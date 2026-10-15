"""XLS-R 300M + AASIST from Tak et al. 2022, ported from fairseq to transformers.

The head repeats SSL_Anti-spoofing/model.py with the same weight names, so the
published checkpoints load as is.
"""

from re import sub

import torch
from torch import nn
from torch.nn.functional import max_pool2d, selu, softmax
from torch.nn.init import xavier_normal_
from transformers import Wav2Vec2Config, Wav2Vec2Model

# fairseq xlsr2_300m.pt. Tak calls it with mask=False, so no SpecAugment here either.
XLSR_300M = Wav2Vec2Config(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
    feat_extract_norm="layer",
    conv_bias=True,
    do_stable_layer_norm=True,
    mask_time_prob=0.0,
)

FAIRSEQ_RENAMES = [
    (r"conv_layers\.(\d)\.0\.", r"conv_layers.\1.conv."),
    (r"conv_layers\.(\d)\.2\.1\.", r"conv_layers.\1.layer_norm."),
    (r"^layer_norm\.", "feature_projection.layer_norm."),
    (r"^post_extract_proj\.", "feature_projection.projection."),
    (r"^encoder\.pos_conv\.0\.", "encoder.pos_conv_embed.conv."),
    (r"\.self_attn\.", ".attention."),
    (r"\.self_attn_layer_norm\.", ".layer_norm."),
    (r"\.fc1\.", ".feed_forward.intermediate_dense."),
    (r"\.fc2\.", ".feed_forward.output_dense."),
]
# Pretraining parts of wav2vec 2.0 that the classifier never runs.
FAIRSEQ_UNUSED = ("mask_emb", "quantizer.", "project_q.", "final_proj.")
POS_CONV = "ssl.encoder.pos_conv_embed.conv."


def _att_weight(dim: int) -> nn.Parameter:
    return nn.Parameter(xavier_normal_(torch.empty(dim, 1)))


def _pairwise(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(2) * x.unsqueeze(1)


def _node_bn(bn: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
    return bn(x.reshape(-1, x.size(-1))).view(x.shape)


class GraphAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, temperature: float):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = _att_weight(out_dim)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(0.2)
        self.temp = temperature

    def forward(self, nodes):
        h = self.input_drop(nodes)
        att = self.att_proj(_pairwise(h)).tanh() @ self.att_weight
        att = softmax(att / self.temp, dim=-2).squeeze(-1)
        out = self.proj_with_att(att @ h) + self.proj_without_att(h)
        return selu(_node_bn(self.bn, out))


class HtrgGraphAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, temperature: float):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        self.att_weight11 = _att_weight(out_dim)
        self.att_weight22 = _att_weight(out_dim)
        self.att_weight12 = _att_weight(out_dim)
        self.att_weightM = _att_weight(out_dim)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(0.2)
        self.temp = temperature

    def _master(self, h, master):
        att = self.att_projM(h * master).tanh() @ self.att_weightM
        att = softmax(att / self.temp, dim=-2).squeeze(-1).unsqueeze(1)
        return self.proj_with_attM(att @ h) + self.proj_without_attM(master)

    def _attention(self, h, n1):
        pair = self.att_proj(_pairwise(h)).tanh()
        att = torch.zeros_like(pair[..., :1])
        att[:, :n1, :n1] = pair[:, :n1, :n1] @ self.att_weight11
        att[:, n1:, n1:] = pair[:, n1:, n1:] @ self.att_weight22
        att[:, :n1, n1:] = pair[:, :n1, n1:] @ self.att_weight12
        att[:, n1:, :n1] = pair[:, n1:, :n1] @ self.att_weight12
        return softmax(att / self.temp, dim=-2).squeeze(-1)

    def forward(self, nodes1, nodes2, master):
        n1 = nodes1.size(1)
        h = torch.cat([self.proj_type1(nodes1), self.proj_type2(nodes2)], dim=1)
        h = self.input_drop(h)
        out = self.proj_with_att(self._attention(h, n1) @ h) + self.proj_without_att(h)
        out = selu(_node_bn(self.bn, out))
        return out[:, :n1], out[:, n1:], self._master(h, master)


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(0.3)

    def forward(self, nodes):
        scores = self.proj(self.drop(nodes)).sigmoid()
        n = max(int(nodes.size(1) * self.k), 1)
        idx = scores.topk(n, dim=1).indices.expand(-1, -1, nodes.size(2))
        return (nodes * scores).gather(1, idx)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, first: bool):
        super().__init__()
        if not first:
            # Trained and saved, but Tak's forward never uses its output.
            self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(2, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=(2, 3), padding=(0, 1))
        self.conv_downsample = nn.Identity()
        if in_ch != out_ch:
            self.conv_downsample = nn.Conv2d(
                in_ch, out_ch, kernel_size=(1, 3), padding=(0, 1)
            )

    def forward(self, x):
        out = self.conv2(selu(self.bn2(self.conv1(x))))
        return out + self.conv_downsample(x)


class SSLAASIST(nn.Module):
    """Takes 16 kHz waveforms (batch, samples), returns logits (spoof, bona fide)."""

    def __init__(self, ssl_config: Wav2Vec2Config = XLSR_300M):
        super().__init__()
        self.ssl = Wav2Vec2Model(ssl_config)
        self.LL = nn.Linear(ssl_config.hidden_size, 128)
        self.first_bn = nn.BatchNorm2d(1)
        self.first_bn1 = nn.BatchNorm2d(64)
        self.drop = nn.Dropout(0.5)
        self.drop_way = nn.Dropout(0.2)
        blocks = []
        for in_ch, out_ch in [
            (1, 32),
            (32, 32),
            (32, 64),
            (64, 64),
            (64, 64),
            (64, 64),
        ]:
            block = ResidualBlock(in_ch, out_ch, first=not blocks)
            blocks.append(nn.Sequential(block))
        self.encoder = nn.Sequential(*blocks)
        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )
        # 42 = 128 LL outputs after the 3x3 max pool.
        self.pos_S = nn.Parameter(torch.randn(1, 42, 64))
        self.master1 = nn.Parameter(torch.randn(1, 1, 64))
        self.master2 = nn.Parameter(torch.randn(1, 1, 64))
        self.GAT_layer_S = GraphAttention(64, 64, temperature=2.0)
        self.GAT_layer_T = GraphAttention(64, 64, temperature=2.0)
        self.HtrgGAT_layer_ST11 = HtrgGraphAttention(64, 32, temperature=100.0)
        self.HtrgGAT_layer_ST12 = HtrgGraphAttention(32, 32, temperature=100.0)
        self.HtrgGAT_layer_ST21 = HtrgGraphAttention(64, 32, temperature=100.0)
        self.HtrgGAT_layer_ST22 = HtrgGraphAttention(32, 32, temperature=100.0)
        self.pool_S = GraphPool(0.5, 64)
        self.pool_T = GraphPool(0.5, 64)
        self.pool_hS1 = GraphPool(0.5, 32)
        self.pool_hT1 = GraphPool(0.5, 32)
        self.pool_hS2 = GraphPool(0.5, 32)
        self.pool_hT2 = GraphPool(0.5, 32)
        self.out_layer = nn.Linear(5 * 32, 2)

    def _branch(self, out_T, out_S, layers, pools, master):
        t, s, m = layers[0](out_T, out_S, master)
        s, t = pools[0](s), pools[1](t)
        t_aug, s_aug, m_aug = layers[1](t, s, m)
        return (
            self.drop_way(t + t_aug),
            self.drop_way(s + s_aug),
            self.drop_way(m + m_aug),
        )

    def forward(self, wav):
        x = self.LL(self.ssl(wav).last_hidden_state)
        x = max_pool2d(x.transpose(1, 2).unsqueeze(1), (3, 3))
        x = selu(self.first_bn(x))
        x = selu(self.first_bn1(self.encoder(x)))
        w = self.attention(x)

        e_S = (x * softmax(w, dim=-1)).sum(-1).transpose(1, 2) + self.pos_S
        out_S = self.pool_S(self.GAT_layer_S(e_S))
        e_T = (x * softmax(w, dim=-2)).sum(-2).transpose(1, 2)
        out_T = self.pool_T(self.GAT_layer_T(e_T))

        t1, s1, m1 = self._branch(
            out_T,
            out_S,
            (self.HtrgGAT_layer_ST11, self.HtrgGAT_layer_ST12),
            (self.pool_hS1, self.pool_hT1),
            self.master1,
        )
        t2, s2, m2 = self._branch(
            out_T,
            out_S,
            (self.HtrgGAT_layer_ST21, self.HtrgGAT_layer_ST22),
            (self.pool_hS2, self.pool_hT2),
            self.master2,
        )
        t, s = torch.maximum(t1, t2), torch.maximum(s1, s2)
        master = torch.maximum(m1, m2).squeeze(1)
        hidden = torch.cat(
            [t.abs().amax(1), t.mean(1), s.abs().amax(1), s.mean(1), master], dim=1
        )
        return self.out_layer(self.drop(hidden))


def hf_key(fairseq_key: str) -> str | None:
    """Tak's checkpoint key to SSLAASIST key, None for weights the model does not have."""
    # The DF checkpoint was saved from DataParallel, the LA one was not.
    key = fairseq_key.removeprefix("module.")
    if not key.startswith("ssl_model.model."):
        return key
    key = key.removeprefix("ssl_model.model.")
    if key.startswith(FAIRSEQ_UNUSED):
        return None
    for pattern, repl in FAIRSEQ_RENAMES:
        key = sub(pattern, repl, key)
    return "ssl." + key


def _parametrized_weight_norm(state: dict) -> dict:
    renamed = dict(state)
    renamed[POS_CONV + "parametrizations.weight.original0"] = renamed.pop(
        POS_CONV + "weight_g"
    )
    renamed[POS_CONV + "parametrizations.weight.original1"] = renamed.pop(
        POS_CONV + "weight_v"
    )
    return renamed


def load_model(path, ssl_config: Wav2Vec2Config = XLSR_300M) -> SSLAASIST:
    """Tak's checkpoint in eval mode; fails if any weight is missing or left over."""
    model = SSLAASIST(ssl_config)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = {}
    for key, value in checkpoint.items():
        new_key = hf_key(key)
        if new_key is not None:
            state[new_key] = value
    # Newer torch keeps weight norm as a parametrization under other names.
    if POS_CONV + "weight_g" not in model.state_dict():
        state = _parametrized_weight_norm(state)
    model.load_state_dict(state)
    return model.eval()

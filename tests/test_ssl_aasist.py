import pytest

pytest.importorskip("transformers")

import torch
from transformers import Wav2Vec2Config

from spoof.ssl_aasist import SSLAASIST, hf_key

TINY = Wav2Vec2Config(
    hidden_size=32,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=64,
    feat_extract_norm="layer",
    conv_bias=True,
    do_stable_layer_norm=True,
    mask_time_prob=0.0,
    num_conv_pos_embeddings=16,
    num_conv_pos_embedding_groups=2,
)
FAIRSEQ = "ssl_model.model."


@pytest.mark.parametrize(
    ("tak", "ours"),
    [
        (
            "feature_extractor.conv_layers.0.0.weight",
            "feature_extractor.conv_layers.0.conv.weight",
        ),
        (
            "feature_extractor.conv_layers.6.2.1.bias",
            "feature_extractor.conv_layers.6.layer_norm.bias",
        ),
        ("layer_norm.weight", "feature_projection.layer_norm.weight"),
        ("post_extract_proj.bias", "feature_projection.projection.bias"),
        ("encoder.pos_conv.0.weight_g", "encoder.pos_conv_embed.conv.weight_g"),
        (
            "encoder.layers.23.self_attn.q_proj.weight",
            "encoder.layers.23.attention.q_proj.weight",
        ),
        (
            "encoder.layers.0.self_attn_layer_norm.bias",
            "encoder.layers.0.layer_norm.bias",
        ),
        (
            "encoder.layers.5.fc1.weight",
            "encoder.layers.5.feed_forward.intermediate_dense.weight",
        ),
        (
            "encoder.layers.5.fc2.bias",
            "encoder.layers.5.feed_forward.output_dense.bias",
        ),
        (
            "encoder.layers.5.final_layer_norm.weight",
            "encoder.layers.5.final_layer_norm.weight",
        ),
        ("encoder.layer_norm.weight", "encoder.layer_norm.weight"),
    ],
)
def test_hf_key_ssl(tak, ours):
    assert hf_key(FAIRSEQ + tak) == "ssl." + ours


@pytest.mark.parametrize(
    "tak", ["mask_emb", "quantizer.vars", "project_q.weight", "final_proj.bias"]
)
def test_hf_key_drops_pretraining_weights(tak):
    assert hf_key(FAIRSEQ + tak) is None


def test_hf_key_keeps_head():
    assert hf_key("HtrgGAT_layer_ST11.att_weightM") == "HtrgGAT_layer_ST11.att_weightM"


def test_forward():
    torch.manual_seed(0)
    model = SSLAASIST(TINY).eval()
    wav = torch.randn(3, 64_600) * 0.1
    with torch.inference_mode():
        logits = model(wav)
        alone = model(wav[1:2])
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits[1], alone[0], atol=1e-4)

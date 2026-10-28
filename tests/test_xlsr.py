import numpy as np
import pytest

pytest.importorskip("transformers")

import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

from spoof.xlsr import length_batches, normalized, pooled_states, with_state_hooks

TINY = Wav2Vec2Config(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=64,
    conv_dim=(16,) * 7,
    feat_extract_norm="layer",
    conv_bias=True,
    do_stable_layer_norm=True,
    mask_time_prob=0.0,
    num_conv_pos_embeddings=16,
    num_conv_pos_embedding_groups=2,
)


def test_length_batches():
    assert length_batches([5, 1, 3, 2, 10], budget=6) == [[1, 3], [2], [0], [4]]


def test_length_batches_empty():
    assert length_batches([], budget=6) == []


def test_normalized():
    wav = np.random.default_rng(0).normal(3, 2, 1000)
    out = normalized(wav)
    assert abs(out.mean()) < 1e-9
    assert out.std() == pytest.approx(1, abs=1e-6)


def test_pooled_states_ignore_padding():
    torch.manual_seed(0)
    model = Wav2Vec2Model(TINY).eval()
    rng = np.random.default_rng(0)
    wavs = [rng.normal(size=n).astype(np.float32) for n in [4000, 6400]]
    together = pooled_states(model, wavs)
    apart = np.concatenate([pooled_states(model, [w]) for w in wavs])
    assert together.shape == (2, TINY.num_hidden_layers + 1, TINY.hidden_size)
    assert np.allclose(together, apart, atol=1e-4)


def replica_states(model, calls: int) -> int:
    """Number of hidden states on the last of `calls` DataParallel-style replica calls."""
    wav = torch.zeros(1, 4000)
    for _ in range(calls):
        replica = model._replicate_for_data_parallel()
        _, _, states = replica(wav, output_hidden_states=True, return_dict=False)
    return len(states)


def test_with_state_hooks_survives_replicas():
    model = with_state_hooks(Wav2Vec2Model(TINY).eval())
    assert replica_states(model, calls=3) == TINY.num_hidden_layers + 1

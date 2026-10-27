import pytest

torch = pytest.importorskip("torch")

from spoof.lcnn import LCNN, LFCC, MaxFeatureMap, _delta


def speech_like(batch, samples):
    torch.manual_seed(0)
    return 0.1 * torch.randn(batch, samples)


def test_lfcc_shape():
    features = LFCC()(speech_like(2, 16_000))
    # Centered frames every 160 samples, 20 coefficients with deltas and delta-deltas.
    assert features.shape == (2, 101, 60)


def test_lfcc_gain_moves_only_energy():
    wav = speech_like(1, 16_000)
    quiet = LFCC()(wav)
    loud = LFCC()(10 * wav)
    assert torch.allclose(loud[:, :, 0], quiet[:, :, 0] + 2, atol=1e-3)
    assert torch.allclose(loud[:, :, 1:], quiet[:, :, 1:], atol=1e-3)


def test_delta():
    ramp = torch.arange(5.0).view(1, 5, 1)
    assert _delta(ramp).flatten().tolist() == [1, 2, 2, 2, 1]


def test_max_feature_map():
    x = torch.tensor([1.0, 5.0, 3.0, 2.0]).view(1, 4, 1, 1)
    assert MaxFeatureMap()(x).flatten().tolist() == [3, 5]


def test_lcnn_scores_any_length():
    model = LCNN().eval()
    with torch.no_grad():
        short = model(speech_like(2, 16_000))
        long = model(speech_like(1, 80_000))
    assert short.shape == (2,)
    assert long.shape == (1,)
    assert torch.isfinite(short).all()

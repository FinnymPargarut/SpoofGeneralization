import numpy as np
import pytest
from soundfile import write

from spoof.audio import (
    SR,
    edge_silence,
    load,
    model_input,
    prepare,
    repeat_crop,
    speech_level,
    to_sr,
)

# librosa trim works in frames of 2048 samples.
FRAME = 2048


def tone(n):
    t = np.arange(n) / SR
    return (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def test_edge_silence():
    wav = np.concatenate([np.zeros(8000), tone(16000), np.zeros(4000)])
    start, end = edge_silence(wav)
    assert 8000 - FRAME <= start <= 8000
    assert 4000 - FRAME <= end <= 4000


def test_edge_silence_without_silence():
    assert edge_silence(tone(16000)) == (0, 0)


def test_edge_silence_all_zeros():
    with pytest.raises(ValueError):
        edge_silence(np.zeros(16000))


def test_repeat_crop():
    long = np.arange(100_000)
    assert np.array_equal(repeat_crop(long), long[:64_600])
    assert np.array_equal(repeat_crop(np.array([1, 2, 3]), 7), [1, 2, 3, 1, 2, 3, 1])
    assert np.array_equal(repeat_crop(np.arange(10), 10), np.arange(10))


def test_repeat_crop_empty():
    with pytest.raises(ValueError):
        repeat_crop(np.array([]))


def test_load(tmp_path):
    path = tmp_path / "a.flac"
    write(path, tone(1600), SR)
    wav = load(path)
    assert wav.dtype == np.float32
    assert np.allclose(wav, tone(1600), atol=1e-4)


@pytest.mark.parametrize(
    ("data", "sr"),
    [
        (tone(1600), 8000),
        (np.stack([tone(1600), tone(1600)], axis=1), SR),
        (np.zeros(0, dtype=np.float32), SR),
    ],
)
def test_load_broken(tmp_path, data, sr):
    path = tmp_path / "a.wav"
    write(path, data, sr)
    with pytest.raises(ValueError):
        load(path)


@pytest.fixture
def files(tmp_path):
    speech = tmp_path / "speech.flac"
    write(speech, tone(16000), SR)
    donor = tmp_path / "donor.flac"
    write(donor, np.linspace(-0.01, 0.01, 8000, dtype=np.float32), SR)
    return speech, donor


def test_model_input_variants(files):
    speech, donor = files
    wav = load(speech)
    orig = model_input({"file_name": speech, "silence": "orig"})
    trim = model_input(
        {"file_name": speech, "silence": "trim", "start": 100, "end": 1100}
    )
    pad = {"file_name": speech, "silence": "pad", "pad_file": donor, "pad_len": 3000}
    assert np.array_equal(orig, repeat_crop(wav))
    assert np.array_equal(trim, repeat_crop(wav[100:1100]))
    expected = repeat_crop(np.concatenate([wav, load(donor)[:3000]]))
    assert np.array_equal(model_input(pad), expected)
    assert np.array_equal(model_input(pad | {"pad_len": 0}), orig)


@pytest.mark.parametrize(
    "variant",
    [
        {"silence": "shift"},
        {"silence": "trim", "start": 20000, "end": 30000},
    ],
)
def test_model_input_broken(files, variant):
    with pytest.raises(ValueError):
        model_input({"file_name": files[0]} | variant)


def test_to_sr():
    sr = 24_000
    t = np.arange(sr) / sr
    stereo = np.stack([np.sin(2 * np.pi * 200 * t)] * 2, axis=1) * 0.5
    wav = to_sr(stereo.astype(np.float32), sr)
    assert wav.dtype == np.float32
    assert wav.shape == (SR,)
    assert 0.45 < np.abs(wav).max() <= 0.55


def test_to_sr_scales_overshoot():
    loud = np.sign(tone(SR)) * 1.5
    wav = to_sr(loud.astype(np.float32), SR)
    assert np.abs(wav).max() == pytest.approx(1)
    assert np.array_equal(np.sign(wav), np.sign(loud))


def test_to_sr_keeps_16k():
    assert np.array_equal(to_sr(tone(1600), SR), tone(1600))


def test_speech_level():
    wav = np.concatenate([np.zeros(8000), tone(16000), np.zeros(8000)])
    # A sine of amplitude 0.5 has power 0.125.
    assert speech_level(wav) == pytest.approx(10 * np.log10(0.125), abs=0.5)


def test_speech_level_silent():
    with pytest.raises(ValueError):
        speech_level(np.zeros(SR))


def test_prepare_trim40_and_gain(tmp_path):
    path = tmp_path / "a.flac"
    wav = np.concatenate([np.zeros(8000), tone(16000), np.zeros(8000)])
    write(path, wav, SR)
    lead, trail = edge_silence(load(path))
    trimmed = prepare({"file_name": path, "silence": "trim40"})
    assert np.array_equal(trimmed, load(path)[lead : wav.size - trail])
    louder = prepare({"file_name": path, "silence": "trim40", "gain_db": 20.0})
    assert np.allclose(louder, 10 * trimmed)

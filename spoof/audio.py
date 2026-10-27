import numpy as np
from librosa import resample
from librosa.effects import trim
from soundfile import read

SR = 16_000


def load(path) -> np.ndarray:
    wav, sr = read(path, dtype="float32")
    if sr != SR or wav.ndim != 1 or wav.size == 0:
        raise ValueError(
            f"{path}: need non-empty mono {SR} Hz, got {sr} Hz, shape {wav.shape}"
        )
    return wav


def to_sr(wav: np.ndarray, sr: int) -> np.ndarray:
    """Mono float32 at SR, scaled down if resampling overshoots full scale."""
    mono = wav.mean(axis=1) if wav.ndim > 1 else wav
    resampled = mono if sr == SR else resample(mono, orig_sr=sr, target_sr=SR)
    # Clipping the overshoot would leave a mark on the loudest files, mostly spoofs.
    peak = np.abs(resampled).max(initial=0)
    scaled = resampled / peak if peak > 1 else resampled
    return scaled.astype(np.float32)


def repeat_crop(wav: np.ndarray, n: int = 64_600) -> np.ndarray:
    """Model input as in Tak's SSL_Anti-spoofing: the first n samples, short audio repeated."""
    if wav.size == 0:
        raise ValueError("empty audio")
    return np.tile(wav, n // wav.size + 1)[:n]


def edge_silence(wav: np.ndarray, top_db: float = 40) -> tuple[int, int]:
    """Leading and trailing silence in samples, librosa trim as in Müller 2021."""
    # librosa measures silence against the loudest frame, so an all-zero signal would have none.
    if not wav.any():
        raise ValueError("audio is all zeros")
    _, (start, end) = trim(wav, top_db=top_db)
    return int(start), int(wav.size - end)


def speech_level(wav: np.ndarray) -> float:
    """Mean power between the edge silences, dB re full scale."""
    lead, trail = edge_silence(wav)
    speech = wav[lead : wav.size - trail]
    return float(10 * np.log10(np.mean(speech**2)))


def _with_silence(wav: np.ndarray, record: dict) -> np.ndarray:
    if record["silence"] == "orig":
        return wav
    if record["silence"] == "trim":
        return wav[record["start"] : record["end"]]
    if record["silence"] == "trim40":
        lead, trail = edge_silence(wav)
        return wav[lead : wav.size - trail]
    if record["silence"] == "pad":
        silence, _ = read(record["pad_file"], frames=record["pad_len"], dtype="float32")
        return np.concatenate([wav, silence])
    raise ValueError(f"unknown silence variant {record['silence']}")


def prepare(record: dict) -> np.ndarray:
    """Full waveform of a protocol row from data.py, silence variant and gain_db applied."""
    wav = _with_silence(load(record["file_name"]), record)
    if "gain_db" in record:
        wav = wav * np.float32(10 ** (record["gain_db"] / 20))
    return wav


def model_input(record: dict) -> np.ndarray:
    """Tak's model input for a protocol row."""
    return repeat_crop(prepare(record))

import numpy as np
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


def _with_silence(wav: np.ndarray, record: dict) -> np.ndarray:
    if record["silence"] == "orig":
        return wav
    if record["silence"] == "trim":
        return wav[record["start"] : record["end"]]
    if record["silence"] == "pad":
        silence, _ = read(record["pad_file"], frames=record["pad_len"], dtype="float32")
        return np.concatenate([wav, silence])
    raise ValueError(f"unknown silence variant {record['silence']}")


def model_input(record: dict) -> np.ndarray:
    """Tak's model input for a protocol row from data.py, silence variant applied."""
    return repeat_crop(_with_silence(load(record["file_name"]), record))

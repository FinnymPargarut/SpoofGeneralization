import numpy as np


def _scores(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("scores must be non-empty and finite")
    return x


def eer(bona, spoof) -> tuple[float, float]:
    """Higher score means bona fide. A trial is accepted when its score is above the threshold."""
    b, s = _scores(bona), _scores(spoof)
    if b.min() > s.max():
        return 0.0, float((b.min() + s.max()) / 2)
    # Only thresholds that can actually split the scores: the official ASVspoof
    # script also splits tied scores and gives EER 1 on constant scores.
    thresholds = np.concatenate([[-np.inf], np.unique(np.concatenate([b, s]))])
    frr = np.searchsorted(np.sort(b), thresholds, side="right") / b.size
    far = (s.size - np.searchsorted(np.sort(s), thresholds, side="right")) / s.size
    i = np.argmin(np.abs(frr - far))
    return float((frr[i] + far[i]) / 2), float(thresholds[i])


def hter(bona, spoof, threshold: float) -> dict[str, float]:
    b, s = _scores(bona), _scores(spoof)
    frr = float(np.mean(b <= threshold))
    far = float(np.mean(s > threshold))
    # Bengio, Mariéthoz 2004: valid only if the threshold was not chosen on these scores.
    sigma = np.sqrt(far * (1 - far) / (4 * s.size) + frr * (1 - frr) / (4 * b.size))
    return {
        "hter": (frr + far) / 2,
        "frr": frr,
        "far": far,
        "ci95": float(1.96 * sigma),
    }

import numpy as np
import pytest
from asvspoof_eval_metrics import compute_eer

from spoof.metrics import eer, hter


@pytest.mark.parametrize(
    ("n_bona", "n_spoof", "seed"),
    [(2548, 22296, 0), (100, 100, 1), (30, 3000, 2), (3000, 30, 3)],
)
def test_eer_matches_official(n_bona, n_spoof, seed):
    rng = np.random.default_rng(seed)
    bona = rng.normal(1.0, 1.0, n_bona)
    spoof = rng.normal(-1.0, 1.5, n_spoof)
    official_eer, official_threshold = compute_eer(bona, spoof)
    assert eer(bona, spoof) == (official_eer, official_threshold)


def test_hter_by_hand():
    bona = [0.1, 0.6, 0.9, 0.95]
    spoof = [0.2, 0.3, 0.7]
    # 0.1 and the tied 0.6 are rejected, 0.7 is accepted.
    result = hter(bona, spoof, 0.6)
    assert result["frr"] == 2 / 4
    assert result["far"] == 1 / 3
    assert result["hter"] == pytest.approx((2 / 4 + 1 / 3) / 2)
    sigma = np.sqrt((1 / 3) * (2 / 3) / (4 * 3) + (2 / 4) * (2 / 4) / (4 * 4))
    assert result["ci95"] == pytest.approx(1.96 * sigma)


@pytest.mark.parametrize("ties", [False, True])
def test_hter_at_eer_threshold_equals_eer(ties):
    rng = np.random.default_rng(4)
    bona = rng.normal(1.0, 1.0, 500)
    spoof = rng.normal(0.0, 1.0, 700)
    if ties:
        bona, spoof = bona.round(), spoof.round()
    error, threshold = eer(bona, spoof)
    assert hter(bona, spoof, threshold)["hter"] == pytest.approx(error)


def test_separated_classes_take_middle_of_gap():
    assert eer([2.0, 3.0], [0.0, 1.0]) == (0.0, 1.5)
    assert hter([2.0, 3.0], [0.0, 1.0], 1.5) == {
        "hter": 0.0,
        "frr": 0.0,
        "far": 0.0,
        "ci95": 0.0,
    }


@pytest.mark.parametrize(
    ("bona", "spoof", "expected"),
    [
        ([1.0], [0.0], 0.0),
        ([0.0], [1.0], 1.0),
        ([0.0, 1.0], [2.0, 3.0], 1.0),
    ],
)
def test_eer_extremes(bona, spoof, expected):
    assert eer(bona, spoof)[0] == expected


def test_constant_scores_give_chance():
    bona, spoof = np.ones(3), np.ones(2)
    assert eer(bona, spoof)[0] == 0.5
    # The official script splits tied scores and reports the worst possible EER.
    assert compute_eer(bona, spoof)[0] == 1.0


@pytest.mark.parametrize("bad", [[], [0.5, np.nan], [0.5, np.inf]])
def test_bad_scores_rejected(bad):
    with pytest.raises(ValueError):
        eer(bad, [0.1, 0.2])
    with pytest.raises(ValueError):
        hter([0.1, 0.2], bad, 0.0)

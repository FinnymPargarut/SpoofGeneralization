import numpy as np
import pandas as pd

from scripts.frozen_lr import pick_layer
from spoof.xlsr import STATES


def test_pick_layer_breaks_zero_eer_ties_by_log_loss():
    protocol = pd.DataFrame({"label": ["bonafide", "bonafide", "spoof", "spoof"]})
    scores = np.zeros((4, STATES), dtype=np.float32)
    scores[:, 3] = [1, 2, -1, -2]
    scores[:, 7] = [5, 6, -5, -6]
    scores[:, 9] = [9, 9, 8, 8]
    assert pick_layer(protocol, scores) == 7


def test_pick_layer_prefers_lower_eer_over_loss():
    protocol = pd.DataFrame({"label": ["bonafide", "bonafide", "spoof", "spoof"]})
    scores = np.zeros((4, STATES), dtype=np.float32)
    scores[:, 2] = [0.1, 0.2, -0.1, -0.2]
    scores[:, 5] = [20, -20, -20, -20]
    assert pick_layer(protocol, scores) == 2

"""Logistic regression on frozen XLS-R features, as in Pascu et al. 2024: C = 1e6, no scaling.

Every layer is fitted on ASVspoof 2019 train; the layer with the lowest dev EER becomes
model xlsr_lr. The last layer fitted on train and dev together repeats Pascu as
xlsr_lr_pascu, for the EER check only: its dev scores are in-sample. Both get a score
CSV and a JSON note per set, the note carries the feature extraction cost. layers.md
has every layer on every set: EER, or FRR at the layer's dev threshold for bona fide
only sets.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from spoof.data import COLUMNS
from spoof.metrics import eer
from spoof.notes import commit
from spoof.sets import SETS
from spoof.xlsr import STATES

LAST = STATES - 1
CHUNK = 8192
FROM_EXTRACTION = [
    "set",
    "silence",
    "files",
    "audio_seconds",
    "seconds",
    "params",
    "device",
    "gpus",
    "gpu0_peak_gb",
]


def fit(x: np.ndarray, is_bona: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Weights, bias and fit seconds. lbfgs gives the same result for any seed."""
    t0 = time.time()
    clf = LogisticRegression(C=1e6, max_iter=5000).fit(x, is_bona)
    return clf.coef_[0], float(clf.intercept_[0]), time.time() - t0


def layer_scores(features: np.ndarray, weights: np.ndarray, bias: np.ndarray):
    """(files, layers) scores, higher is bona fide; features are read in chunks."""
    scores = np.empty((len(features), len(bias)), dtype=np.float32)
    for start in range(0, len(features), CHUNK):
        chunk = np.asarray(features[start : start + CHUNK], dtype=np.float32)
        scores[start : start + CHUNK] = np.einsum("nlh,lh->nl", chunk, weights) + bias
    return scores


def classes(protocol: pd.DataFrame, scores: np.ndarray):
    is_bona = (protocol["label"] == "bonafide").to_numpy()
    return scores[is_bona], scores[~is_bona]


def layer_cell(protocol: pd.DataFrame, scores: np.ndarray, threshold: float) -> str:
    bona, spoof = classes(protocol, scores)
    if spoof.size:
        return f"{100 * eer(bona, spoof)[0]:.2f}"
    return f"FRR {100 * np.mean(bona <= threshold):.2f}"


def layers_table(protocols: dict, scores: dict, best: int) -> list[str]:
    names = [n for n in protocols if n != "train"]
    lines = [
        "| layer | " + " | ".join(names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]
    for layer in range(STATES):
        dev_bona, dev_spoof = classes(protocols["dev"], scores["dev"][:, layer])
        threshold = eer(dev_bona, dev_spoof)[1]
        cells = [f"{layer}, dev best" if layer == best else str(layer)]
        for name in names:
            cells.append(layer_cell(protocols[name], scores[name][:, layer], threshold))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def save(model: str, protocols: dict, scores: dict, features_dir: Path, info: dict):
    """Score CSV and note per set except train, train.json with the training cost.

    info goes into every note; training cost is the fit plus feature extraction of
    the sets the model was fitted on.
    """
    out = info["out"] / model
    out.mkdir(parents=True, exist_ok=True)
    extraction_seconds = 0.0
    for name in info["trained_on"].split("+"):
        extraction_seconds += json.loads((features_dir / f"{name}.json").read_text())[
            "seconds"
        ]
    training = {
        "model": model,
        "trained_on": info["trained_on"],
        "fit_seconds": info["fit_seconds"],
        "extraction_seconds": extraction_seconds,
        "seconds": round(info["fit_seconds"] + extraction_seconds, 1),
        "commit": commit(),
    }
    (out / "train.json").write_text(json.dumps(training, indent=2))
    for name, protocol in protocols.items():
        if name == "train":
            continue
        scored = protocol[COLUMNS].assign(score=scores[name])
        scored.to_csv(out / f"{name}.csv", index=False)
        extraction = json.loads((features_dir / f"{name}.json").read_text())
        note = {"model": model, "commit": commit()}
        for key in ["layer", "trained_on", "trainable", "fit_seconds"]:
            note[key] = info[key]
        for key in FROM_EXTRACTION:
            if key in extraction:
                note[key] = extraction[key]
        (out / f"{name}.json").write_text(json.dumps(note, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, required=True, help="xlsr_features.py output"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    names = [n for n in SETS if (args.features / f"{n}.json").exists()]
    protocols = {}
    features = {}
    for name in names:
        csv = args.features / f"{name}.csv"
        protocols[name] = pd.read_csv(csv, keep_default_na=False)
        features[name] = np.load(args.features / f"{name}.npy", mmap_mode="r")
    train = np.asarray(features["train"], dtype=np.float32)
    train_bona = (protocols["train"]["label"] == "bonafide").to_numpy()
    hidden = train.shape[2]

    weights = np.zeros((STATES, hidden))
    bias = np.zeros(STATES)
    fit_seconds = np.zeros(STATES)
    for layer in range(STATES):
        weights[layer], bias[layer], fit_seconds[layer] = fit(
            train[:, layer], train_bona
        )
        print(f"layer {layer}: fitted in {fit_seconds[layer]:.0f} s", flush=True)
    scores = {}
    for name in names:
        scores[name] = layer_scores(features[name], weights, bias)
    dev_errors = []
    for layer in range(STATES):
        dev_errors.append(eer(*classes(protocols["dev"], scores["dev"][:, layer]))[0])
    best = int(np.argmin(dev_errors))

    args.out.mkdir(parents=True, exist_ok=True)
    table = layers_table(protocols, scores, best)
    (args.out / "layers.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))

    info = {
        "out": args.out,
        "trainable": hidden + 1,
        "layer": best,
        "trained_on": "train",
        "fit_seconds": round(fit_seconds[best], 1),
    }
    best_scores = {}
    for name in names:
        best_scores[name] = scores[name][:, best]
    save("xlsr_lr", protocols, best_scores, args.features, info)

    dev = np.asarray(features["dev"][:, LAST], dtype=np.float32)
    dev_bona = (protocols["dev"]["label"] == "bonafide").to_numpy()
    x = np.concatenate([train[:, LAST], dev])
    w, b, seconds = fit(x, np.concatenate([train_bona, dev_bona]))
    pascu_scores = {}
    for name in names:
        last = features[name][:, LAST : LAST + 1]
        pascu_scores[name] = layer_scores(last, w[None], np.array([b]))[:, 0]
    info["layer"] = LAST
    info["trained_on"] = "train+dev"
    info["fit_seconds"] = round(seconds, 1)
    save("xlsr_lr_pascu", protocols, pascu_scores, args.features, info)


if __name__ == "__main__":
    main()

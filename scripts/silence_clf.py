"""Silence shortcut check on CPU: classifiers that see only silence and loudness.

Müller 2021: leading silence by librosa trim at 40 dB, MLP 128x128, trained on
ASVspoof 2019 train. The same MLP on leading and trailing silence and loudness
also runs on In-the-Wild with speaker-disjoint folds. Per-file features are saved
next to the results and reused on a rerun.
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spoof.audio import SR, edge_silence, load
from spoof.data import asvspoof2019, read_in_the_wild
from spoof.metrics import eer

SEEDS = [0, 1, 2]
FOLDS = 5
FEATURES = {
    "lead": ["lead"],
    "lead, trail, loudness": ["lead", "trail", "loudness"],
}


def features(path: str) -> tuple[float, float, float]:
    """Edge silence in seconds and loudness of the rest in dB re full scale."""
    wav = load(path)
    lead, trail = edge_silence(wav)
    speech = wav[lead : wav.size - trail]
    return lead / SR, trail / SR, 10 * np.log10(np.mean(speech**2))


def extract(args) -> pd.DataFrame:
    parts = []
    for split in ["train", "dev", "eval"]:
        parts.append(asvspoof2019(args.asvspoof, split).assign(set=split))
    itw = read_in_the_wild(args.itw / "meta.csv", args.itw)
    parts.append(itw.assign(set="itw"))
    files = pd.concat(parts, ignore_index=True)
    print(f"features of {len(files)} files on {args.workers} processes", flush=True)
    with Pool(args.workers) as pool:
        values = np.array(pool.map(features, files["file_name"], chunksize=64))
    return files.assign(lead=values[:, 0], trail=values[:, 1], loudness=values[:, 2])


def load_or_extract(path: Path, args) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, keep_default_na=False)
    files = extract(args)
    files.to_csv(path, index=False)
    return files


def fit(train: pd.DataFrame, columns: list[str], seed: int):
    mlp = MLPClassifier(hidden_layer_sizes=(128, 128), random_state=seed)
    return make_pipeline(StandardScaler(), mlp).fit(
        train[columns], train["label"] == "bonafide"
    )


def error(clf, test: pd.DataFrame, columns: list[str]) -> float:
    scores = clf.predict_proba(test[columns])[:, 1]
    is_bona = (test["label"] == "bonafide").to_numpy()
    return eer(scores[is_bona], scores[~is_bona])[0]


def asvspoof_runs(files: pd.DataFrame, columns: list[str]) -> dict[str, list[float]]:
    train = files[files["set"] == "train"]
    runs = {"dev": [], "eval": []}
    for seed in SEEDS:
        clf = fit(train, columns, seed)
        for split, errors in runs.items():
            errors.append(error(clf, files[files["set"] == split], columns))
    return runs


def itw_runs(files: pd.DataFrame, columns: list[str]) -> list[float]:
    itw = files[files["set"] == "itw"].reset_index(drop=True)
    folds = list(GroupKFold(FOLDS).split(itw, groups=itw["speaker"]))
    errors = []
    for seed in SEEDS:
        for train_idx, test_idx in folds:
            clf = fit(itw.iloc[train_idx], columns, seed)
            errors.append(error(clf, itw.iloc[test_idx], columns))
    return errors


def line(cells: list) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def summary(errors: list[float]) -> str:
    return f"{100 * np.mean(errors):.2f} ± {100 * np.std(errors):.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asvspoof", type=Path, required=True, help="folder with ASVspoof2019_LA_*"
    )
    parser.add_argument(
        "--itw", type=Path, required=True, help="folder with meta.csv and wav"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    files = load_or_extract(args.out / "silence_features.csv", args)
    lines = [
        line(["features", "test set", "EER, % (mean ± std)", "runs"]),
        line(["---"] * 4),
    ]
    for name, columns in FEATURES.items():
        for split, errors in asvspoof_runs(files, columns).items():
            lines.append(
                line([name, f"ASVspoof 2019 {split}", summary(errors), len(errors)])
            )
        errors = itw_runs(files, columns)
        test_set = f"In-the-Wild, {FOLDS} speaker folds"
        lines.append(line([name, test_set, summary(errors), len(errors)]))
    lines += ["", "Müller 2021, lead only: dev 31.09, eval 15.12 (mean of 4 runs)."]
    print("\n".join(lines))
    (args.out / "silence_clf.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

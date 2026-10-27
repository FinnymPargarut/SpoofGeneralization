"""Shortcut check on CPU: classifiers that see only simple properties of a recording.

Müller 2021: leading silence by librosa trim at 40 dB, MLP 128x128, trained on
ASVspoof 2019 train. The same MLP on leading and trailing silence and loudness
also runs on In-the-Wild with speaker-disjoint folds. With --stage2, RuASD gets the
same MLP on silence and loudness, duration and the original sample rate, in random
stratified folds: RuASD has no speaker labels. Per-file features are saved next to
the results and reused on a rerun.
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spoof.audio import SR, edge_silence, load, speech_level
from spoof.data import asvspoof2019, read_in_the_wild
from spoof.metrics import eer
from spoof.sets import protocol

SEEDS = [0, 1, 2]
FOLDS = 5
FEATURES = {
    "lead": ["lead"],
    "lead, trail, loudness": ["lead", "trail", "loudness"],
}
RUASD_FEATURES = {
    "lead, trail, loudness": ["lead", "trail", "loudness"],
    "duration": ["duration"],
    "sample rate": ["sr"],
    "all five": ["lead", "trail", "loudness", "duration", "sr"],
}


def features(path: str) -> tuple[float, float, float, float]:
    """Edge silence and duration in seconds, loudness of the rest in dB re full scale."""
    wav = load(path)
    lead, trail = edge_silence(wav)
    return lead / SR, trail / SR, speech_level(wav), wav.size / SR


def with_features(files: pd.DataFrame, workers: int) -> pd.DataFrame:
    print(f"features of {len(files)} files on {workers} processes", flush=True)
    with Pool(workers) as pool:
        values = np.array(pool.map(features, files["file_name"], chunksize=64))
    return files.assign(
        lead=values[:, 0],
        trail=values[:, 1],
        loudness=values[:, 2],
        duration=values[:, 3],
    )


def stage1_files(args) -> pd.DataFrame:
    parts = []
    for split in ["train", "dev", "eval"]:
        parts.append(asvspoof2019(args.asvspoof, split).assign(set=split))
    itw = read_in_the_wild(args.itw / "meta.csv", args.itw)
    parts.append(itw.assign(set="itw"))
    return pd.concat(parts, ignore_index=True)


def load_or_extract(path: Path, files, workers: int) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, keep_default_na=False)
    done = with_features(files(), workers)
    done.to_csv(path, index=False)
    return done


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


def fold_runs(files: pd.DataFrame, columns: list[str], folds_of) -> list[float]:
    """folds_of(seed) gives (train, test) index pairs over files."""
    errors = []
    for seed in SEEDS:
        for train_idx, test_idx in folds_of(seed):
            clf = fit(files.iloc[train_idx], columns, seed)
            errors.append(error(clf, files.iloc[test_idx], columns))
    return errors


def line(cells: list) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def summary(errors: list[float]) -> str:
    return f"{100 * np.mean(errors):.2f} ± {100 * np.std(errors):.2f}"


def stage1_lines(args) -> list[str]:
    files = load_or_extract(
        args.out / "silence_features.csv", lambda: stage1_files(args), args.workers
    )
    itw = files[files["set"] == "itw"].reset_index(drop=True)
    speaker_folds = list(GroupKFold(FOLDS).split(itw, groups=itw["speaker"]))
    lines = []
    for name, columns in FEATURES.items():
        for split, errors in asvspoof_runs(files, columns).items():
            lines.append(
                line([name, f"ASVspoof 2019 {split}", summary(errors), len(errors)])
            )
        errors = fold_runs(itw, columns, lambda seed: speaker_folds)
        test_set = f"In-the-Wild, {FOLDS} speaker folds"
        lines.append(line([name, test_set, summary(errors), len(errors)]))
    return lines


def ruasd_lines(args) -> list[str]:
    ruasd = protocol("ruasd", args)
    files = load_or_extract(
        args.out / "ruasd_features.csv", lambda: ruasd, args.workers
    )

    def stratified(seed):
        folds = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
        return folds.split(files, files["label"])

    lines = []
    for name, columns in RUASD_FEATURES.items():
        errors = fold_runs(files, columns, stratified)
        test_set = f"RuASD sample, {FOLDS} random folds"
        lines.append(line([name, test_set, summary(errors), len(errors)]))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asvspoof", type=Path, help="folder with ASVspoof2019_LA_*")
    parser.add_argument("--itw", type=Path, help="folder with meta.csv and wav")
    parser.add_argument("--stage2", type=Path, help="unpacked fetch_stage2.py output")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    lines = [
        line(["features", "test set", "EER, % (mean ± std)", "runs"]),
        line(["---"] * 4),
    ]
    if args.stage2:
        lines += ruasd_lines(args)
    if args.asvspoof:
        lines += stage1_lines(args)
        lines += ["", "Müller 2021, lead only: dev 31.09, eval 15.12 (mean of 4 runs)."]
    print("\n".join(lines))
    (args.out / "silence_clf.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

"""Protocols of all sets by name, shared by the scoring and feature scripts."""

from pathlib import Path

import pandas as pd

from spoof.data import (
    asvspoof2019,
    read_etrim,
    read_in_the_wild,
    read_local,
    with_pad,
    with_trim,
)

# Made by fetch_stage2.py: a protocol CSV and an unpacked folder of the same name.
LOCAL = ["ruasd", "librispeech", "ruls"]
SETS = [
    "train",
    "dev",
    "eval",
    "eval_trim",
    "eval_pad",
    "itw",
    "itw_trim40",
    "ruasd",
    "librispeech",
    "librispeech_trim40",
    "ruls",
    "ruls_trim40",
]


def _eval_pad(asvspoof: Path, etrim: Path, seed: int) -> pd.DataFrame:
    train = asvspoof2019(asvspoof, "train")
    donors = with_trim(train[train["label"] == "bonafide"], read_etrim(etrim))
    return with_pad(asvspoof2019(asvspoof, "eval"), donors, seed)


def protocol(name: str, paths, seed: int = 0) -> pd.DataFrame:
    """paths holds folders asvspoof, itw, stage2 and the LA19etrim log etrim.

    A set needs only its own inputs, the rest may be None. seed picks eval_pad donors.
    """
    base = name.removesuffix("_trim40")
    if base != name:
        return protocol(base, paths, seed).assign(silence="trim40")
    if name in ["train", "dev", "eval"]:
        return asvspoof2019(paths.asvspoof, name)
    if name == "eval_trim":
        return with_trim(asvspoof2019(paths.asvspoof, "eval"), read_etrim(paths.etrim))
    if name == "eval_pad":
        return _eval_pad(paths.asvspoof, paths.etrim, seed)
    if name == "itw":
        return read_in_the_wild(paths.itw / "meta.csv", paths.itw)
    if name in LOCAL:
        return read_local(paths.stage2 / f"{name}.csv", paths.stage2)
    raise ValueError(f"unknown set {name}")

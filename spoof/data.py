from pathlib import Path

import numpy as np
import pandas as pd

# file_name and label as in Speech DF Arena protocols, the rest are ours.
COLUMNS = ["file_name", "label", "attack", "speaker", "silence"]


def _check(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    protocol = df[COLUMNS].reset_index(drop=True)
    if protocol.empty or protocol.isna().any().any():
        raise ValueError("protocol is empty or has missing fields")
    unknown = set(protocol["label"]) - {"bonafide", "spoof"}
    if unknown:
        raise ValueError(f"unknown labels: {sorted(unknown)}")
    return protocol


def read_asvspoof2019(protocol: Path, audio_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        protocol,
        sep=" ",
        header=None,
        names=["speaker", "name", "env", "attack", "label"],
        dtype=str,
    )
    audio = Path(audio_dir)
    df["file_name"] = [str(audio / f"{name}.flac") for name in df["name"]]
    df["silence"] = "orig"
    return _check(df)


def asvspoof2019(la_dir: Path, split: str) -> pd.DataFrame:
    """la_dir holds the ASVspoof2019_LA_* folders as in the official LA.zip."""
    kind = "trn" if split == "train" else "trl"
    protocols = Path(la_dir) / "ASVspoof2019_LA_cm_protocols"
    return read_asvspoof2019(
        protocols / f"ASVspoof2019.LA.cm.{split}.{kind}.txt",
        Path(la_dir) / f"ASVspoof2019_LA_{split}" / "flac",
    )


def read_in_the_wild(meta_csv: Path, audio_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(meta_csv, usecols=["file", "speaker", "label"], dtype=str)
    audio = Path(audio_dir)
    df["file_name"] = [str(audio / name) for name in df["file"]]
    df["label"] = df["label"].replace({"bona-fide": "bonafide"})
    df["attack"] = "-"
    df["silence"] = "orig"
    return _check(df)


def read_protocol(path: Path) -> pd.DataFrame:
    return _check(pd.read_csv(path, dtype=str))


def read_etrim(log: Path) -> pd.DataFrame:
    df = pd.read_csv(log, sep=" ", usecols=["TRIAL", "START_INDEX", "END_INDEX"])
    df = df.rename(
        columns={"TRIAL": "name", "START_INDEX": "start", "END_INDEX": "end"}
    )
    if df.empty:
        raise ValueError("no trim marks")
    df[["start", "end"]] = df[["start", "end"]].astype(int)
    if (df["start"] >= df["end"]).any():
        raise ValueError("trim marks with start >= end")
    return df[["name", "start", "end"]]


def with_trim(protocol: pd.DataFrame, marks: pd.DataFrame) -> pd.DataFrame:
    """Adds speech start and end from LA19etrim marks, every file must have them."""
    names = [Path(f).stem for f in protocol["file_name"]]
    found = marks.set_index("name").reindex(names)
    if found.isna().any().any():
        raise ValueError("some files have no trim marks")
    return protocol.assign(
        silence="trim",
        start=found["start"].to_numpy(dtype=int),
        end=found["end"].to_numpy(dtype=int),
    )


def with_pad(protocol: pd.DataFrame, donors: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Appends to each file the leading silence of a random donor from with_trim.

    Donors are drawn with replacement, so silence lengths follow the donors' own
    distribution, zero length included.
    """
    rng = np.random.default_rng(seed)
    picked = donors.iloc[rng.integers(len(donors), size=len(protocol))]
    return protocol.assign(
        silence="pad",
        pad_file=picked["file_name"].to_numpy(),
        pad_len=picked["start"].to_numpy(),
    )

"""Stage 1 tables from score.py output: EER per set, HTER at the dev threshold.

The threshold is the EER point on ASVspoof 2019 dev. HTER at the eval threshold
is for comparison with Zhou, Wang 2026.
"""

import argparse
from pathlib import Path

import pandas as pd

from spoof.metrics import eer, hter

ORDER = ["dev", "eval", "eval_trim", "eval_pad", "itw"]
# Different data with original silence; trim and pad are variants of eval.
MEAN_OVER = ["eval", "itw"]
HEADER = [
    "set",
    "bona",
    "spoof",
    "EER",
    "HTER ± 95%",
    "FRR",
    "FAR",
    "HTER, eval thr",
    "FRR, eval thr",
]


def classes(scores: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    bona = scores.loc[scores["label"] == "bonafide", "score"]
    spoof = scores.loc[scores["label"] == "spoof", "score"]
    return bona, spoof


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def row(cells: list) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def table(header: list[str], rows: list[list]) -> list[str]:
    lines = [row(header), row(["---"] * len(header))]
    for cells in rows:
        lines.append(row(cells))
    return lines


def dev_note(bona: pd.Series, spoof: pd.Series, error: float, threshold: float) -> str:
    note = f"Threshold at dev EER point: {threshold:.4f}, dev EER {pct(error)}%."
    if error > 0:
        return note
    gap = bona.min() - spoof.max()
    return note + f" Classes are separated on dev, gap {gap:.4f}."


def set_rows(sets: dict[str, pd.DataFrame], dev_thr: float, eval_thr: float) -> list:
    rows = []
    for name, scores in sets.items():
        bona, spoof = classes(scores)
        at_dev = hter(bona, spoof, dev_thr)
        at_eval = hter(bona, spoof, eval_thr)
        rows.append(
            [
                name,
                len(bona),
                len(spoof),
                pct(eer(bona, spoof)[0]),
                f"{pct(at_dev['hter'])} ± {pct(at_dev['ci95'])}",
                pct(at_dev["frr"]),
                pct(at_dev["far"]),
                pct(at_eval["hter"]),
                pct(at_eval["frr"]),
            ]
        )
    return rows


def attack_rows(sets: dict[str, pd.DataFrame], variants: list[str]) -> list:
    attacks = sorted(set(sets["eval"]["attack"]) - {"-"})
    rows = []
    for attack in attacks:
        cells = [attack]
        for name in variants:
            scores = sets[name]
            bona, _ = classes(scores)
            spoof = scores.loc[scores["attack"] == attack, "score"]
            cells.append(pct(eer(bona, spoof)[0]))
        rows.append(cells)
    return rows


def report(model: str, sets: dict[str, pd.DataFrame]) -> list[str]:
    dev_bona, dev_spoof = classes(sets["dev"])
    dev_eer, dev_thr = eer(dev_bona, dev_spoof)
    _, eval_thr = eer(*classes(sets["eval"]))
    errors = []
    for name in MEAN_OVER:
        errors.append(eer(*classes(sets[name]))[0])
    mean_eer = pct(sum(errors) / len(errors))
    mean_over = ", ".join(MEAN_OVER)
    lines = [f"## {model}", "", dev_note(dev_bona, dev_spoof, dev_eer, dev_thr), ""]
    lines += table(HEADER, set_rows(sets, dev_thr, eval_thr))
    variants = [name for name in sets if name.startswith("eval")]
    lines += [
        "",
        "All in %. HTER on dev and eval-threshold columns on eval are in-sample.",
        "In-the-Wild interval is too narrow: its spoofs are cut from 219 videos.",
        f"Mean EER over {mean_over}: {mean_eer}%.",
        "",
        "EER by attack, %",
        "",
    ]
    lines += table(["attack", *variants], attack_rows(sets, variants))
    return lines


def read_sets(folder: Path) -> dict[str, pd.DataFrame]:
    sets = {}
    for name in ORDER:
        path = folder / f"{name}.csv"
        if path.exists():
            sets[name] = pd.read_csv(path, keep_default_na=False)
    return sets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, help="score.py output folder")
    args = parser.parse_args()
    lines = []
    for folder in sorted(args.scores.iterdir()):
        if (folder / "dev.csv").exists():
            lines += report(folder.name, read_sets(folder)) + [""]
    print("\n".join(lines))


if __name__ == "__main__":
    main()

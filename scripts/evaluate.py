"""Result tables from score folders: EER per set, HTER at the dev threshold.

Usage: evaluate.py outputs/stage1 outputs/stage2. A model folder present in several
stage folders is merged, later folders win. The threshold is the EER point on
ASVspoof 2019 dev; HTER at the eval threshold is for comparison with Zhou, Wang 2026.
Bona fide only sets get FRR at the dev threshold, overall and per speaker. Models
named <name>_seed<k> also get a summary over seeds, every model a cost row.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from spoof.metrics import eer, hter
from spoof.sets import SETS

MEAN_OVER = ["eval", "itw", "ruasd"]
HEADER = [
    "set",
    "language",
    "bona",
    "spoof",
    "EER",
    "HTER ± 95%",
    "FRR",
    "FAR",
    "HTER, eval thr",
    "FRR, eval thr",
]
BONA_HEADER = [
    "set",
    "language",
    "files",
    "speakers",
    "FRR",
    "FRR by speaker, mean ± std",
    "median score",
]
COST_HEADER = [
    "model",
    "params, M",
    "trainable",
    "training, h",
    "inference, s per hour of audio",
    "files per s",
    "GPU0 peak, GB",
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


def language(scores: pd.DataFrame) -> str:
    # Files without the column are ASVspoof and In-the-Wild, both English.
    return scores["language"].iloc[0] if "language" in scores else "en"


def dev_note(bona: pd.Series, spoof: pd.Series, error: float, threshold: float) -> str:
    note = f"Threshold at dev EER point: {threshold:.4f}, dev EER {pct(error)}%."
    if error > 0:
        return note
    gap = bona.min() - spoof.max()
    return note + f" Classes are separated on dev, gap {gap:.4f}."


def at_eval(bona, spoof, eval_thr) -> list[str]:
    if eval_thr is None:
        return ["-", "-"]
    result = hter(bona, spoof, eval_thr)
    return [pct(result["hter"]), pct(result["frr"])]


def set_rows(sets: dict, dev_thr: float, eval_thr) -> list:
    rows = []
    for name, scores in sets.items():
        bona, spoof = classes(scores)
        if spoof.empty:
            continue
        at_dev = hter(bona, spoof, dev_thr)
        rows.append(
            [
                name,
                language(scores),
                len(bona),
                len(spoof),
                pct(eer(bona, spoof)[0]),
                f"{pct(at_dev['hter'])} ± {pct(at_dev['ci95'])}",
                pct(at_dev["frr"]),
                pct(at_dev["far"]),
                *at_eval(bona, spoof, eval_thr),
            ]
        )
    return rows


def bona_rows(sets: dict, dev_thr: float) -> list:
    rows = []
    for name, scores in sets.items():
        if (scores["label"] == "spoof").any():
            continue
        rejected = scores["score"] <= dev_thr
        by_speaker = rejected.groupby(scores["speaker"]).mean()
        rows.append(
            [
                name,
                language(scores),
                len(scores),
                len(by_speaker),
                pct(rejected.mean()),
                f"{pct(by_speaker.mean())} ± {pct(by_speaker.std(ddof=0))}",
                f"{scores['score'].median():.3f}",
            ]
        )
    return rows


def attack_rows(sets: dict, variants: list[str]) -> list:
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


def ruasd_lines(scores: pd.DataFrame, dev_thr: float) -> list[str]:
    """EER by generator against all bona fide, FRR by bona fide corpus."""
    bona, _ = classes(scores)
    systems = []
    for attack, part in scores[scores["label"] == "spoof"].groupby("attack"):
        far = (part["score"] > dev_thr).mean()
        systems.append([attack, len(part), pct(eer(bona, part["score"])[0]), pct(far)])
    systems.sort(key=lambda cells: -float(cells[2]))
    corpora = []
    for corpus, part in scores[scores["label"] == "bonafide"].groupby("attack"):
        frr = (part["score"] <= dev_thr).mean()
        corpora.append([corpus, len(part), pct(frr), f"{part['score'].median():.3f}"])
    lines = ["", "RuASD, EER by system against all bona fide, %", ""]
    lines += table(["system", "files", "EER", "FAR at dev thr"], systems)
    lines += ["", "RuASD, bona fide by corpus at the dev threshold", ""]
    lines += table(["corpus", "files", "FRR, %", "median score"], corpora)
    return lines


def report(model: str, sets: dict) -> list[str]:
    dev_bona, dev_spoof = classes(sets["dev"])
    dev_eer, dev_thr = eer(dev_bona, dev_spoof)
    eval_thr = eer(*classes(sets["eval"]))[1] if "eval" in sets else None
    present = [name for name in MEAN_OVER if name in sets]
    errors = []
    for name in present:
        errors.append(eer(*classes(sets[name]))[0])
    lines = [f"## {model}", "", dev_note(dev_bona, dev_spoof, dev_eer, dev_thr), ""]
    lines += table(HEADER, set_rows(sets, dev_thr, eval_thr))
    lines += [
        "",
        "All in %. HTER on dev and eval-threshold columns on eval are in-sample.",
    ]
    if "itw" in sets:
        lines.append(
            "In-the-Wild interval is too narrow: its spoofs are cut from 219 videos."
        )
    if errors:
        mean = pct(np.mean(errors))
        lines.append(f"Mean EER over {', '.join(present)}: {mean}%.")
    bona_only = bona_rows(sets, dev_thr)
    if bona_only:
        lines += ["", "Bona fide only sets, FRR at the dev threshold, %", ""]
        lines += table(BONA_HEADER, bona_only)
    variants = [name for name in sets if name.startswith("eval")]
    if "eval" in sets:
        lines += ["", "EER by attack, %", ""]
        lines += table(["attack", *variants], attack_rows(sets, variants))
    if "ruasd" in sets:
        lines += ruasd_lines(sets["ruasd"], dev_thr)
    return lines


def seed_lines(base: str, models: dict) -> list[str]:
    """Mean ± std over seeds of EER and HTER at each seed's own dev threshold."""
    names = [n for n in SETS if all(n in sets for sets in models.values())]
    rows = []
    for name in names:
        errors = []
        hters = []
        for sets in models.values():
            dev_thr = eer(*classes(sets["dev"]))[1]
            bona, spoof = classes(sets[name])
            if spoof.empty:
                break
            errors.append(eer(bona, spoof)[0])
            hters.append(hter(bona, spoof, dev_thr)["hter"])
        if errors:
            rows.append(
                [
                    name,
                    f"{pct(np.mean(errors))} ± {pct(np.std(errors))}",
                    f"{pct(np.mean(hters))} ± {pct(np.std(hters))}",
                ]
            )
    title = f"## {base}, {len(models)} seeds, mean ± std, %"
    return [title, ""] + table(["set", "EER", "HTER"], rows)


def notes(folder: Path) -> list[dict]:
    found = []
    for path in sorted(folder.glob("*.json")):
        found.append(json.loads(path.read_text()))
    return found


def trained_on_dev(folders: list[Path]) -> bool:
    for folder in folders:
        for note in notes(folder):
            if "dev" in note.get("trained_on", ""):
                return True
    return False


def cost_row(model: str, folders: list[Path]) -> list:
    set_notes = []
    training = {}
    for folder in folders:
        for note in notes(folder):
            if "set" in note:
                set_notes.append(note)
            else:
                training = note
    params = training.get("params", 0)
    for note in set_notes:
        params = max(params, note.get("params", 0))
    trainable = training.get("params", "-")
    if set_notes and "trainable" in set_notes[0]:
        trainable = set_notes[0]["trainable"]
    hours = f"{training['seconds'] / 3600:.2f}" if "seconds" in training else "-"
    timed = [n for n in set_notes if "audio_seconds" in n]
    timed_seconds = sum(n["seconds"] for n in timed)
    audio = sum(n["audio_seconds"] for n in timed)
    seconds = sum(n["seconds"] for n in set_notes)
    files = sum(n["files"] for n in set_notes)
    peaks = [n["gpu0_peak_gb"] for n in set_notes if "gpu0_peak_gb" in n]
    return [
        model,
        f"{params / 1e6:.1f}" if params else "-",
        trainable,
        hours,
        f"{timed_seconds / audio * 3600:.0f}" if audio else "-",
        f"{files / seconds:.1f}" if seconds else "-",
        max(peaks) if peaks else "-",
    ]


def read_models(stages: list[Path]) -> tuple[dict, dict]:
    """Scores by model and set, and every folder that holds the model."""
    models = {}
    folders = {}
    for stage in stages:
        for folder in sorted(p for p in stage.iterdir() if p.is_dir()):
            for path in sorted(folder.glob("*.csv")):
                scores = pd.read_csv(path, keep_default_na=False)
                models.setdefault(folder.name, {})[path.stem] = scores
            folders.setdefault(folder.name, []).append(folder)
    return models, folders


def in_set_order(sets: dict) -> dict:
    ordered = {}
    for name in SETS:
        if name in sets:
            ordered[name] = sets[name]
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", type=Path, nargs="+", help="stage output folders")
    args = parser.parse_args()
    models, folders = read_models(args.stages)
    scored = {}
    for model, sets in models.items():
        if "dev" in sets and "score" in sets["dev"]:
            scored[model] = in_set_order(sets)
    lines = []
    for model, sets in scored.items():
        lines += report(model, sets)
        if trained_on_dev(folders[model]):
            lines.append(
                "Fitted on train and dev: the dev threshold is in-sample, EER only."
            )
        lines.append("")
    by_base = {}
    for model, sets in scored.items():
        match = re.fullmatch(r"(.+)_seed\d+", model)
        if match:
            by_base.setdefault(match.group(1), {})[model] = sets
    for base, seeds in by_base.items():
        if len(seeds) > 1:
            lines += seed_lines(base, seeds) + [""]
    cost_rows = []
    for model in scored:
        cost_rows.append(cost_row(model, folders[model]))
    lines += ["## Cost", ""] + table(COST_HEADER, cost_rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

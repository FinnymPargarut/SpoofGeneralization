"""Scores of a Tak SSL-AASIST checkpoint, a CSV and a JSON note per set.

Score is the bona fide logit as in Tak's code: higher means bona fide.
Sets already scored in the output folder are skipped.
"""

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from spoof.audio import model_input
from spoof.data import (
    COLUMNS,
    asvspoof2019,
    read_etrim,
    read_in_the_wild,
    with_pad,
    with_trim,
)
from spoof.ssl_aasist import load_model

SETS = ["dev", "eval", "eval_trim", "eval_pad", "itw"]


def protocols(args) -> dict[str, pd.DataFrame]:
    marks = read_etrim(args.etrim)
    train = asvspoof2019(args.asvspoof, "train")
    donors = with_trim(train[train["label"] == "bonafide"], marks)
    eval_ = asvspoof2019(args.asvspoof, "eval")
    return {
        "dev": asvspoof2019(args.asvspoof, "dev"),
        "eval": eval_,
        "eval_trim": with_trim(eval_, marks),
        "eval_pad": with_pad(eval_, donors, args.seed),
        "itw": read_in_the_wild(args.itw / "meta.csv", args.itw),
    }


def collate(records: list[dict]) -> torch.Tensor:
    return torch.from_numpy(np.stack([model_input(r) for r in records]))


def on_all_gpus(model: torch.nn.Module) -> torch.nn.Module:
    if not torch.cuda.is_available():
        return model
    on_gpu = model.cuda()
    if torch.cuda.device_count() > 1:
        return torch.nn.DataParallel(on_gpu)
    return on_gpu


def device_note() -> dict:
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    return {
        "device": torch.cuda.get_device_name(0),
        "gpus": torch.cuda.device_count(),
        "gpu0_peak_gb": round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
    }


def commit() -> str:
    repo = Path(__file__).resolve().parent.parent
    git = ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"]
    return subprocess.run(
        git, check=False, capture_output=True, text=True
    ).stdout.strip()


def score(model, protocol: pd.DataFrame, args, name: str) -> np.ndarray:
    loader = DataLoader(
        protocol.to_dict("records"),
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    device = next(model.parameters()).device
    started = time.time()
    scores = []
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            logits = model(batch.to(device, non_blocking=True))
            scores.append(logits[:, 1].float().cpu().numpy())
            if i % 100 == 0:
                done = min((i + 1) * args.batch_size, len(protocol))
                elapsed = time.time() - started
                print(f"{name}: {done}/{len(protocol)}, {elapsed:.0f} s", flush=True)
    return np.concatenate(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True, help="output subfolder")
    parser.add_argument(
        "--asvspoof", type=Path, required=True, help="folder with ASVspoof2019_LA_*"
    )
    parser.add_argument(
        "--itw", type=Path, required=True, help="folder with meta.csv and wav"
    )
    parser.add_argument("--etrim", type=Path, required=True, help="LA2019-sil-trim.log")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sets", nargs="+", choices=SETS, default=SETS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="donors for eval_pad")
    args = parser.parse_args()

    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in args.sets if not (out / f"{s}.csv").exists()]
    model = on_all_gpus(load_model(args.checkpoint))
    for name, protocol in protocols(args).items():
        if name not in todo:
            continue
        started = datetime.now(UTC).isoformat(timespec="seconds")
        t0 = time.time()
        scores = score(model, protocol, args, name)
        seconds = round(time.time() - t0, 1)
        protocol[COLUMNS].assign(score=scores).to_csv(out / f"{name}.csv", index=False)
        note = {
            "model": "ssl_aasist",
            "checkpoint": args.checkpoint.name,
            "set": name,
            "silence": protocol["silence"].iloc[0],
            "files": len(protocol),
            "commit": commit(),
            "started": started,
            "seconds": seconds,
            "batch_size": args.batch_size,
            "pad_seed": args.seed,
        }
        note.update(device_note())
        (out / f"{name}.json").write_text(json.dumps(note, indent=2))
        print(f"{name}: done in {seconds} s")


if __name__ == "__main__":
    main()

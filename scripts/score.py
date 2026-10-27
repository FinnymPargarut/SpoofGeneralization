"""Scores of a model on protocol sets, a CSV and a JSON note per set.

ssl_aasist: Tak's checkpoint on the first 4 s, the bona fide logit as in Tak's code.
lcnn: model.pt from train_lcnn.py on the whole recording, one trial per batch.
Higher score means bona fide. Sets already scored in the output folder are skipped.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from spoof.audio import SR, prepare, repeat_crop
from spoof.data import COLUMNS
from spoof.lcnn import LCNN
from spoof.notes import commit, device_note, on_all_gpus
from spoof.sets import SETS, protocol
from spoof.ssl_aasist import load_model


def cropped(records: list[dict]) -> tuple[torch.Tensor, int]:
    wavs = [prepare(r) for r in records]
    batch = np.stack([repeat_crop(w) for w in wavs])
    return torch.from_numpy(batch), sum(w.size for w in wavs)


def whole(records: list[dict]) -> tuple[torch.Tensor, int]:
    (record,) = records
    wav = prepare(record)
    return torch.from_numpy(wav).unsqueeze(0), wav.size


def load(name: str, checkpoint: Path) -> torch.nn.Module:
    if name == "ssl_aasist":
        return on_all_gpus(load_model(checkpoint))
    model = LCNN()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.cuda().eval() if torch.cuda.is_available() else model.eval()


def bona_fide(name: str, output: torch.Tensor) -> torch.Tensor:
    return output[:, 1] if name == "ssl_aasist" else output


def score(model, protocol_: pd.DataFrame, args, name: str) -> tuple[np.ndarray, int]:
    lcnn = args.model == "lcnn"
    loader = DataLoader(
        protocol_.to_dict("records"),
        batch_size=1 if lcnn else args.batch_size,
        num_workers=args.workers,
        collate_fn=whole if lcnn else cropped,
        pin_memory=torch.cuda.is_available(),
    )
    device = next(model.parameters()).device
    started = time.time()
    scores = []
    samples = 0
    with torch.inference_mode():
        for i, (batch, batch_samples) in enumerate(loader):
            output = model(batch.to(device, non_blocking=True))
            scores.append(bona_fide(args.model, output).float().cpu().numpy())
            samples += batch_samples
            if i % 1000 == 0:
                elapsed = time.time() - started
                print(f"{name}: batch {i}/{len(loader)}, {elapsed:.0f} s", flush=True)
    return np.concatenate(scores), samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["ssl_aasist", "lcnn"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True, help="output subfolder")
    parser.add_argument("--sets", nargs="+", choices=SETS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--asvspoof", type=Path, help="folder with ASVspoof2019_LA_*")
    parser.add_argument("--itw", type=Path, help="folder with meta.csv and wav")
    parser.add_argument("--etrim", type=Path, help="LA2019-sil-trim.log")
    parser.add_argument("--stage2", type=Path, help="unpacked fetch_stage2.py output")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="donors for eval_pad")
    args = parser.parse_args()

    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in args.sets if not (out / f"{s}.csv").exists()]
    if not todo:
        return
    model = load(args.model, args.checkpoint)
    params = sum(p.numel() for p in model.parameters())
    for name in todo:
        protocol_ = protocol(name, args, args.seed)
        started = datetime.now(UTC).isoformat(timespec="seconds")
        t0 = time.time()
        scores, samples = score(model, protocol_, args, name)
        seconds = round(time.time() - t0, 1)
        protocol_[COLUMNS].assign(score=scores).to_csv(out / f"{name}.csv", index=False)
        note = {
            "model": args.model,
            "checkpoint": args.checkpoint.name,
            "params": params,
            "set": name,
            "silence": protocol_["silence"].iloc[0],
            "files": len(protocol_),
            "audio_seconds": round(samples / SR, 1),
            "commit": commit(),
            "started": started,
            "seconds": seconds,
            "batch_size": 1 if args.model == "lcnn" else args.batch_size,
            "seed": args.seed,
        }
        note.update(device_note())
        (out / f"{name}.json").write_text(json.dumps(note, indent=2))
        print(f"{name}: done in {seconds} s")


if __name__ == "__main__":
    main()

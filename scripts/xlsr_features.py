"""Frozen XLS-R 300M features of protocol sets, whole recordings.

Per set: <set>.npy of float16 (files, 25, 1024) time-averaged hidden states, the
protocol <set>.csv in the same row order and a JSON note. Sets with a note in the
output folder are skipped.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
from soundfile import info
from torch.utils.data import DataLoader

from spoof.audio import SR, prepare
from spoof.notes import commit, device_note, on_all_gpus
from spoof.sets import SETS, protocol
from spoof.xlsr import NAME, STATES, length_batches, load_xlsr, pooled_states


def frames(path: str) -> int:
    return info(path).frames


def load_group(records: list[dict], indices: list[int]):
    wavs = []
    for i in indices:
        wavs.append(prepare(records[i]))
    return indices, wavs


def extract(model, hidden: int, records: list[dict], out: Path, args) -> int:
    # Headers are read in parallel: one at a time is slow on the Kaggle input disk.
    with ThreadPool(32) as pool:
        lengths = pool.map(frames, [r["file_name"] for r in records])
    loader = DataLoader(
        range(len(records)),
        batch_sampler=length_batches(lengths, args.budget * SR),
        collate_fn=partial(load_group, records),
        num_workers=args.workers,
    )
    features = np.lib.format.open_memmap(
        out, mode="w+", dtype=np.float16, shape=(len(records), STATES, hidden)
    )
    samples = 0
    started = time.time()
    for i, (indices, wavs) in enumerate(loader):
        features[indices] = pooled_states(model, wavs)
        samples += sum(w.size for w in wavs)
        if i % 200 == 0:
            print(
                f"{out.stem}: batch {i}/{len(loader)}, {time.time() - started:.0f} s",
                flush=True,
            )
    features.flush()
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sets", nargs="+", choices=SETS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--asvspoof", type=Path, help="folder with ASVspoof2019_LA_*")
    parser.add_argument("--itw", type=Path, help="folder with meta.csv and wav")
    parser.add_argument("--etrim", type=Path, help="LA2019-sil-trim.log")
    parser.add_argument("--stage2", type=Path, help="unpacked fetch_stage2.py output")
    parser.add_argument(
        "--budget", type=int, default=160, help="seconds of padded audio per batch"
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in args.sets if not (args.out / f"{s}.json").exists()]
    if not todo:
        return
    xlsr = load_xlsr()
    hidden = xlsr.config.hidden_size
    params = sum(p.numel() for p in xlsr.parameters())
    model = on_all_gpus(xlsr)
    for name in todo:
        protocol_ = protocol(name, args)
        started = datetime.now(UTC).isoformat(timespec="seconds")
        t0 = time.time()
        records = protocol_.to_dict("records")
        samples = extract(model, hidden, records, args.out / f"{name}.npy", args)
        protocol_.to_csv(args.out / f"{name}.csv", index=False)
        note = {
            "model": NAME,
            "set": name,
            "silence": protocol_["silence"].iloc[0],
            "files": len(protocol_),
            "params": params,
            "audio_seconds": round(samples / SR, 1),
            "commit": commit(),
            "started": started,
            "seconds": round(time.time() - t0, 1),
            "budget_seconds": args.budget,
        }
        note.update(device_note())
        (args.out / f"{name}.json").write_text(json.dumps(note, indent=2))
        print(f"{name}: done in {note['seconds']} s")


if __name__ == "__main__":
    main()

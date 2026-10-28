"""LFCC-LCNN training on ASVspoof 2019 LA train, recipe of Wang, Yamagishi 2021.

Adam 3e-4 halved every 10 epochs, batches of 64 trials of similar length padded with
zeros as in the NII code. The epoch with the lowest dev loss is saved as model.pt
with a JSON note; training stops after `patience` epochs without a better one.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from soundfile import read
from torch.nn.functional import binary_cross_entropy_with_logits

from spoof.lcnn import LCNN
from spoof.notes import commit, device_note
from spoof.sets import protocol


def pcm(path: str) -> np.ndarray:
    wav, _ = read(path, dtype="int16")
    return wav


def load_split(args, split: str):
    protocol_ = protocol(split, args)
    with Pool(args.workers) as pool:
        wavs = pool.map(pcm, protocol_["file_name"], chunksize=64)
    labels = (protocol_["label"] == "bonafide").to_numpy(dtype=np.float32)
    return wavs, labels


def similar_length_batches(lengths, batch_size: int, rng) -> list[np.ndarray]:
    """NII block shuffle: sort by length, shuffle inside blocks of 4 batches, shuffle blocks."""
    order = np.argsort(lengths, kind="stable")
    block = 4 * batch_size
    blocks = []
    for start in range(0, len(order), block):
        blocks.append(rng.permutation(order[start : start + block]))
    rng.shuffle(blocks)
    flat = np.concatenate(blocks)
    return np.array_split(flat, np.arange(batch_size, len(flat), batch_size))


def padded(wavs: list[np.ndarray]) -> torch.Tensor:
    batch = np.zeros((len(wavs), max(w.size for w in wavs)), dtype=np.float32)
    for i, wav in enumerate(wavs):
        batch[i, : wav.size] = wav / 32768
    return torch.from_numpy(batch)


def batch_loss(model, wavs, labels, idx, device) -> torch.Tensor:
    x = padded([wavs[i] for i in idx]).to(device)
    y = torch.from_numpy(labels[idx]).to(device)
    return binary_cross_entropy_with_logits(model(x), y)


def dev_loss(model, wavs, labels, batch_size, device) -> float:
    order = np.argsort([w.size for w in wavs], kind="stable")
    total = 0.0
    model.eval()
    with torch.inference_mode():
        for idx in np.array_split(order, np.arange(batch_size, len(order), batch_size)):
            total += batch_loss(model, wavs, labels, idx, device).item() * len(idx)
    return total / len(order)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asvspoof", type=Path, required=True, help="folder with ASVspoof2019_LA_*"
    )
    parser.add_argument("--out", type=Path, required=True, help="model folder")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "train.json").exists():
        return
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_wavs, train_labels = load_split(args, "train")
    dev_wavs, dev_labels = load_split(args, "dev")
    lengths = [w.size for w in train_wavs]

    model = LCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    t0 = time.time()
    best_loss, best_epoch, epoch = np.inf, -1, -1
    for epoch in range(args.epochs):
        model.train()
        for idx in similar_length_batches(lengths, args.batch_size, rng):
            loss = batch_loss(model, train_wavs, train_labels, idx, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        loss_dev = dev_loss(model, dev_wavs, dev_labels, args.batch_size, device)
        print(f"epoch {epoch}: dev loss {loss_dev:.4f}, {time.time() - t0:.0f} s")
        if loss_dev < best_loss:
            best_loss, best_epoch = loss_dev, epoch
            torch.save(model.state_dict(), args.out / "model.pt")
        elif epoch - best_epoch >= args.patience:
            break
    note = {
        "model": "lcnn",
        "seed": args.seed,
        "params": sum(p.numel() for p in model.parameters()),
        "epochs": epoch + 1,
        "best_epoch": best_epoch,
        "best_dev_loss": round(best_loss, 5),
        "train_files": len(train_wavs),
        "commit": commit(),
        "started": started,
        "seconds": round(time.time() - t0, 1),
    }
    note.update(device_note())
    (args.out / "train.json").write_text(json.dumps(note, indent=2))


if __name__ == "__main__":
    main()

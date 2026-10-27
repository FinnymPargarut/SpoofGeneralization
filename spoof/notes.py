"""Run notes written next to scores: code version and hardware."""

import subprocess
from pathlib import Path

import torch


def commit() -> str:
    repo = Path(__file__).resolve().parent.parent
    git = ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"]
    return subprocess.run(
        git, check=False, capture_output=True, text=True
    ).stdout.strip()


def device_note() -> dict:
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    return {
        "device": torch.cuda.get_device_name(0),
        "gpus": torch.cuda.device_count(),
        "gpu0_peak_gb": round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
    }


def on_all_gpus(model: torch.nn.Module) -> torch.nn.Module:
    if not torch.cuda.is_available():
        return model
    on_gpu = model.cuda()
    if torch.cuda.device_count() > 1:
        return torch.nn.DataParallel(on_gpu)
    return on_gpu

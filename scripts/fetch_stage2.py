"""Stage 2 inputs, run once on a Kaggle CPU session with internet on.

RuASD: every tenth record of each system and corpus, picked by a hash of its name,
from the shards with original records; shards are streamed and only picked records kept.
LibriSpeech dev-clean and test-clean, RuLS: the first PER_SPEAKER recordings of each
speaker in archive order. Each set becomes <set>.zip of 16 kHz FLAC and <set>.csv,
a protocol with file names relative to the unpacked zip. Sets with a CSV are skipped.
"""

import argparse
import io
import json
import tarfile
import time
from collections import Counter
from hashlib import md5
from http.client import HTTPException
from multiprocessing import Pool
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import pandas as pd
from soundfile import info, read, write

from spoof.audio import SR, speech_level, to_sr
from spoof.data import COLUMNS

RUASD = "https://huggingface.co/datasets/lab260/RuASD/resolve/main/ruasd-{:06d}.tar"
# 0-192 hold the original records, 193-249 their augmented copies.
RUASD_SHARDS = range(193)
LIBRISPEECH = [
    "https://openslr.elda.org/resources/12/dev-clean.tar.gz",
    "https://openslr.elda.org/resources/12/test-clean.tar.gz",
]
RULS = ["https://openslr.elda.org/resources/96/ruls_data.tar.gz"]
PER_SPEAKER = 100
# Median speech level of ASVspoof 2019 train bona fide: the level detectors know.
TARGET_DB = -17.0
RETRIES = 3


def picked(sample_id: str) -> bool:
    return int(md5(sample_id.encode()).hexdigest(), 16) % 10 == 0


def flac(wav: np.ndarray) -> bytes:
    buf = io.BytesIO()
    write(buf, wav, SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def remote_tar(url: str, mode: str):
    """(tar, member) pairs of a streamed archive; read a member before the next one."""
    with (
        urlopen(url, timeout=300) as response,
        tarfile.open(fileobj=response, mode=mode) as tar,
    ):
        for member in tar:
            yield tar, member


def ruasd_record(meta: dict, data: bytes):
    """Protocol row and FLAC bytes, None for empty or silent audio."""
    wav, sr = read(io.BytesIO(data), dtype="float32")
    if wav.size == 0 or not wav.any():
        return None
    if meta["label"] not in ["real", "fake"]:
        raise ValueError(f"{meta['sample_id']}: unknown label {meta['label']}")
    row = {
        "file_name": f"ruasd/{meta['sample_id']}.flac",
        "label": "spoof" if meta["label"] == "fake" else "bonafide",
        # The generator of a spoof, the source corpus of a bona fide recording.
        "attack": meta["subset"] or meta["audio_relpath"].split("/")[2],
        "speaker": "-",
        "language": "ru",
        "silence": "orig",
        "sr": sr,
        "subtype": info(io.BytesIO(data)).subtype,
        "seconds": round(len(wav) / sr, 3),
    }
    return row, flac(to_sr(wav, sr))


def read_ruasd_shard(shard: int):
    """Kept rows and FLAC files, count of silent records, bytes of audio and metadata read."""
    rows = []
    files = []
    skipped = 0
    size = 0
    meta = None
    for tar, member in remote_tar(RUASD.format(shard), "r|"):
        size += member.size
        stem, _, ext = member.name.rpartition(".")
        if ext == "json":
            meta = json.load(tar.extractfile(member))
            continue
        if meta is None or meta["sample_id"] != stem or not picked(stem):
            continue
        record = ruasd_record(meta, tar.extractfile(member).read())
        if record is None:
            skipped += 1
            continue
        rows.append(record[0])
        files.append((record[0]["file_name"], record[1]))
    return rows, files, skipped, size


def ruasd_shard(shard: int):
    """A failed shard comes back empty, so one bad shard does not lose hours of others."""
    for attempt in range(1, RETRIES + 1):
        t0 = time.time()
        try:
            rows, files, skipped, size = read_ruasd_shard(shard)
            return shard, rows, files, skipped, size, time.time() - t0, False
        except (OSError, HTTPException, tarfile.TarError) as error:
            print(f"shard {shard}, attempt {attempt}: {error!r}", flush=True)
            time.sleep(30 * attempt)
    return shard, [], [], 0, 0, 0.0, True


def ruasd(out: Path, processes: int) -> dict:
    rows = []
    skipped = 0
    downloaded = 0
    failed = []
    t0 = time.time()
    with (
        ZipFile(out / "ruasd.zip", "w", ZIP_STORED) as archive,
        Pool(processes) as pool,
    ):
        for result in pool.imap_unordered(ruasd_shard, RUASD_SHARDS):
            shard, shard_rows, files, shard_skipped, size, seconds, error = result
            if error:
                failed.append(shard)
            for name, data in files:
                archive.writestr(name, data)
            rows += shard_rows
            skipped += shard_skipped
            downloaded += size
            speed = downloaded / (time.time() - t0) / 1e6
            print(
                f"shard {shard}: {len(shard_rows)} kept in {seconds:.0f} s, "
                f"{downloaded / 1e9:.1f} GB at {speed:.1f} MB/s overall",
                flush=True,
            )
    protocol = pd.DataFrame(rows).sort_values("file_name")
    protocol.to_csv(out / "ruasd.csv", index=False)
    counts = Counter(protocol["attack"])
    return {
        "files": len(protocol),
        "skipped_silent": skipped,
        "failed_shards": sorted(failed),
        "per_source": counts,
    }


def speaker_of(name: str, ext: str) -> str | None:
    """Speaker folder of LibriSpeech/<split>/<speaker>/... and <split>/audio/<reader>/..."""
    parts = name.split("/")
    return parts[2] if name.endswith(ext) and len(parts) > 3 else None


def librivox_row(
    file_name: str, speaker: str, language: str, audio: np.ndarray
) -> dict:
    return {
        "file_name": file_name,
        "label": "bonafide",
        "attack": "-",
        "speaker": speaker,
        "language": language,
        "silence": "orig",
        "gain_db": round(TARGET_DB - speech_level(audio), 2),
        "seconds": round(audio.size / SR, 3),
    }


def librivox(out: Path, name: str, urls: list[str], ext: str, language: str) -> dict:
    counts = Counter()
    rows = []
    skipped = 0
    with ZipFile(out / f"{name}.zip", "w", ZIP_STORED) as archive:
        for url in urls:
            for tar, member in remote_tar(url, "r|gz"):
                speaker = speaker_of(member.name, ext)
                if speaker is None or counts[speaker] >= PER_SPEAKER:
                    continue
                data = tar.extractfile(member).read()
                wav, sr = read(io.BytesIO(data), dtype="float32")
                audio = to_sr(wav, sr)
                if not audio.any():
                    skipped += 1
                    continue
                file_name = f"{name}/{speaker}_{Path(member.name).stem}.flac"
                archive.writestr(file_name, flac(audio))
                rows.append(librivox_row(file_name, speaker, language, audio))
                counts[speaker] += 1
    columns = [*COLUMNS, "gain_db", "seconds"]
    pd.DataFrame(rows, columns=columns).to_csv(out / f"{name}.csv", index=False)
    return {"files": len(rows), "skipped_silent": skipped, "per_speaker": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    names = ["librispeech", "ruls", "ruasd"]
    parser.add_argument("--sets", nargs="+", choices=names, default=names)
    parser.add_argument("--processes", type=int, default=8, help="RuASD shards at once")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.sets:
        if (args.out / f"{name}.csv").exists():
            continue
        t0 = time.time()
        if name == "ruasd":
            note = ruasd(args.out, args.processes)
        elif name == "librispeech":
            note = librivox(args.out, name, LIBRISPEECH, ".flac", "en")
        else:
            note = librivox(args.out, name, RULS, ".wav", "ru")
        note["seconds"] = round(time.time() - t0)
        (args.out / f"{name}.json").write_text(json.dumps(note, indent=2))
        print(f"{name}: {note['files']} files in {note['seconds']} s", flush=True)


if __name__ == "__main__":
    main()

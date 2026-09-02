"""Extract features for the whole library, in parallel.

Decoding and analysing a track takes a few seconds, so the collection is
spread across cores. Results are cached per track: re-running only touches
tracks whose features are missing or older than the audio file.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import features, library, paths




def _job(args):
    track_id, path, beats, offset, phrases = args
    try:
        feats = features.analyse(path, beats, offset, phrases)
        if not feats:
            return track_id, "empty", 0
        features.save(paths.features_dir() / f"{track_id}.npz", feats)
        return track_id, "ok", len(feats)
    except Exception as exc:                       # noqa: BLE001
        return track_id, f"{type(exc).__name__}: {exc}"[:90], 0


def collect(db_path: str | None = None, force: bool = False) -> list:
    con = library.connect(db_path)
    jobs = []
    for r in con.execute("""SELECT id, path FROM tracks
                            WHERE analyzed=1 AND present=1 ORDER BY id"""):
        cache = paths.features_dir() / f"{r['id']}.npz"
        if not force and cache.exists():
            try:
                if cache.stat().st_mtime >= os.path.getmtime(r["path"]):
                    continue
            except OSError:
                pass
        beats = np.array([x[0] for x in con.execute(
            "SELECT time_ms FROM beats WHERE track_id=? ORDER BY idx", (r["id"],))])
        nums = [x[0] for x in con.execute(
            "SELECT number FROM beats WHERE track_id=? ORDER BY idx", (r["id"],))]
        offset = next((i for i, n in enumerate(nums) if n == 1), 0)
        phrases = [(x[0], x[1], x[2]) for x in con.execute(
            "SELECT idx,start_beat,end_beat FROM phrases WHERE track_id=? ORDER BY idx",
            (r["id"],))]
        if len(beats) and phrases:
            jobs.append((r["id"], r["path"], beats, offset, phrases))
    con.close()
    return jobs


def run(db_path: str | None = None, force: bool = False,
        workers: int | None = None) -> None:
    paths.ensure()
    jobs = collect(db_path, force)
    if not jobs:
        print("All features are up to date.")
        return

    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 4) - 2))
    print(f"Analysing {len(jobs)} tracks across {workers} workers...")

    ok = 0
    failures = []
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for i, (tid, status, n) in enumerate(
                pool.imap_unordered(_job, jobs, chunksize=1), 1):
            if status == "ok":
                ok += 1
            else:
                failures.append((tid, status))
            done = time.time() - t0
            eta = done / i * (len(jobs) - i)
            print(f"\r  {i}/{len(jobs)}  ok={ok}  failed={len(failures)}  "
                  f"eta {eta:>4.0f}s", end="", flush=True)
    print(f"\nDone in {time.time()-t0:.0f}s -- {ok} succeeded, "
          f"{len(failures)} failed.")
    for tid, why in failures[:10]:
        print(f"  track {tid}: {why}")


if __name__ == "__main__":
    run(force="--force" in sys.argv)

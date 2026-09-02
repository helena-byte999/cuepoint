"""Recognising which record is playing, and where the needle is.

Neither Serato nor rekordbox will tell us what is on the deck. Serato has no
public API at all, and rekordbox only broadcasts over Pro DJ Link when real
Pioneer hardware is on the network. So Cuepoint does not ask the DJ software:
it listens to the room and works it out.

That sounds like Shazam, but it is a far easier problem. Shazam identifies
arbitrary audio against tens of millions of tracks. We only ever need to know
which of a few hundred *known* records is playing -- a closed set, already
analysed. Closed-set matching can use a much richer representation than a
sparse hash, and it gives us something a hash cannot:

    cross-correlating the live chroma against the stored chroma returns the
    best offset as well as the best track, and that offset is the playhead.

Track plus playhead is everything downstream needs. It tells us which phrase
is currently sounding, and `mix.recommend` turns that into what to play next
and how many bars are left to decide.

What defeats it
---------------
Pitch. A DJ moves the fader +/-8%, so live audio runs at a different rate to
the index and a long window smears the correlation. We test several rate
hypotheses and keep the best. With keylock on -- the modern default -- pitch
classes are preserved and chroma stays valid; with keylock off a large pitch
move shifts energy across chroma bins and confidence will drop.

Repetition, which is worse. Loop-based music genuinely repeats: in this
collection an eight-second window at bar 32 is not merely similar to bar 64,
it is indistinguishable -- measured, and adding a spectral-shape band to the
feature changed the result by nothing at all, because the timbre repeats too.
Longer windows do not rescue it either (8s/16s/24s/32s all land within noise
of each other). So a single window cannot fix the playhead, and pretending
otherwise would put the needle a whole section out.

What it can do is name the record, which it does reliably. Position is
therefore not decided by one observation but carried: `match` returns several
candidate offsets, the listener advances them all on a clock, and the wrong
ones die as soon as the arrangement moves -- the breakdown arrives at bar 64
and not at bar 32, and only the surviving hypothesis predicts it. Identity is
immediate; position converges over a few seconds.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from . import audio, features as featmod, library, paths



# Coarser than the analysis in features.py: identity needs time resolution, not
# pitch precision, and a smaller index keeps the whole library correlatable in
# well under a second.
NFFT, HOP = 4096, 2048          # 10.8 frames/sec at 22.05 kHz
FPS = audio.SAMPLE_RATE / HOP

# Rate hypotheses, covering the pitch fader's range. The live signal is
# resampled by each in turn and the best score kept.
RATES = (0.94, 0.97, 1.0, 1.03, 1.06)

# Chroma frames are non-negative and unit-normalised, so cosine similarity is
# high even between unrelated records -- the floor is not 0, it is around 0.83.
# Measured over the collection: a correct match scores 0.967 at its 5th
# percentile, while the best a wrong record ever managed was 0.908. The
# threshold sits in that gap. Setting it by intuition instead of measurement
# puts it far too low and every wrong answer reads as confident.
MIN_CONFIDENCE = 0.94
MIN_MARGIN = 0.02        # ...and it must beat the runner-up by this much


@dataclass
class Match:
    track_id: int
    offset_ms: int       # playhead at the END of the query window
    confidence: float    # 0..1, mean cosine similarity over the window
    margin: float        # lead over the best rival track
    rate: float          # rate hypothesis that won, ~= deck pitch

    @property
    def solid(self) -> bool:
        return self.confidence >= MIN_CONFIDENCE and self.margin >= MIN_MARGIN


# --------------------------------------------------------------------------
# building the index
# --------------------------------------------------------------------------

def chroma_series(y: np.ndarray) -> np.ndarray:
    """Per-frame chroma for a signal, L2-normalised frame by frame.

    Normalising per frame is what makes this work through a microphone in a
    loud room: it throws away level and keeps only the shape of the pitch
    content, so the match does not care how loud the speakers are.
    """
    spec = audio.spectrogram(y, NFFT, HOP)
    chroma = featmod._chroma_matrix(audio.fft_freqs(NFFT)) @ spec   # (12, n)
    chroma = chroma.T.astype(np.float32)                            # (n, 12)
    norm = np.linalg.norm(chroma, axis=1, keepdims=True)
    return np.divide(chroma, norm, out=np.zeros_like(chroma), where=norm > 0)


def _job(args):
    track_id, path = args
    try:
        return track_id, chroma_series(audio.decode(path)), ""
    except Exception as exc:                       # noqa: BLE001
        return track_id, None, f"{type(exc).__name__}: {exc}"[:90]


def build(db_path: str | None = None, out: Path | None = None,
          workers: int | None = None) -> None:
    """Decode the library once and store every track's chroma over time."""
    out = Path(out) if out is not None else paths.index_path()
    paths.ensure()
    con = library.connect(db_path)
    jobs = [(r["id"], r["path"]) for r in con.execute(
        "SELECT id, path FROM tracks WHERE analyzed=1 AND present=1 ORDER BY id")]
    con.close()
    if not jobs:
        print("Nothing to index -- run `build` and `extract` first.")
        return

    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 4) - 2))
    print(f"Indexing {len(jobs)} tracks across {workers} workers...")

    frames, owner, starts = [], [], {}
    n = 0
    failed = []
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for i, (tid, series, err) in enumerate(
                pool.imap_unordered(_job, jobs, chunksize=1), 1):
            if series is None or len(series) == 0:
                failed.append((tid, err or "empty"))
            else:
                starts[tid] = n
                frames.append(series)
                owner.append(np.full(len(series), tid, dtype=np.int64))
                n += len(series)
            print(f"\r  {i}/{len(jobs)}  frames={n}", end="", flush=True)

    all_frames = np.concatenate(frames).astype(np.float16)   # half the memory
    np.savez_compressed(
        out, frames=all_frames, owner=np.concatenate(owner),
        track_ids=np.array(list(starts)), offsets=np.array(list(starts.values())),
        fps=np.array([FPS]),
    )
    size = out.stat().st_size / 1e6
    print(f"\nIndexed {len(starts)} tracks, {n} frames, {size:.1f} MB "
          f"in {time.time() - t0:.0f}s")
    for tid, why in failed[:5]:
        print(f"  track {tid}: {why}")


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------

class Index:
    """The chroma index, held in memory and ready to correlate against."""

    def __init__(self, path: Path | None = None):
        path = Path(path) if path is not None else paths.index_path()
        with np.load(path) as z:
            self.frames = z["frames"].astype(np.float32)   # (N, 12)
            self.owner = z["owner"]                        # track id per frame
            self.fps = float(z["fps"][0])
        # Where each track begins, so a global offset becomes a position.
        self.starts: dict[int, int] = {}
        change = np.flatnonzero(np.diff(self.owner)) + 1
        for i in np.concatenate([[0], change]):
            self.starts[int(self.owner[i])] = int(i)
        self.n_tracks = len(self.starts)

    @staticmethod
    def _retime(q: np.ndarray, rate: float) -> np.ndarray:
        """Resample a query along time, to test a deck-pitch hypothesis."""
        if rate == 1.0:
            return q
        n = max(2, int(round(len(q) / rate)))
        src = np.arange(len(q))
        dst = np.linspace(0, len(q) - 1, n)
        return np.stack([np.interp(dst, src, q[:, d]) for d in range(12)], axis=1
                        ).astype(np.float32)

    def _score(self, q: np.ndarray) -> np.ndarray:
        """Mean cosine similarity of the query at every offset in the index.

        Done as a correlation per chroma dimension and summed. Correlating in
        the Fourier domain keeps this linear-ish in library size, and avoids
        ever materialising the (N x window) similarity matrix, which for a
        real library would be hundreds of megabytes.
        """
        n_q = len(q)
        if n_q < 4 or n_q > len(self.frames):
            return np.zeros(0, dtype=np.float32)
        total = np.zeros(len(self.frames) - n_q + 1, dtype=np.float32)
        for d in range(12):
            total += fftconvolve(self.frames[:, d], q[::-1, d], mode="valid")
        return total / n_q

    def match_local(self, query: np.ndarray, track_id: int, centre_ms: int,
                    radius_ms: int = 6000, rate: float = 1.0) -> Match | None:
        """Confirm a position we already believe, cheaply.

        Once a record is locked there is no reason to search the whole library
        again -- and every reason not to, since a global search re-opens the
        repetition ambiguity that continuity has already resolved. This looks
        only in a window around where the clock says we are.
        """
        start = self.starts.get(track_id)
        if start is None:
            return None
        end = start + int(np.sum(self.owner == track_id))
        q = self._retime(query, rate)
        n_q = len(q)

        centre = start + int(centre_ms / 1000.0 * self.fps) - n_q
        lo = max(start, centre - int(radius_ms / 1000.0 * self.fps))
        hi = min(end - n_q, centre + int(radius_ms / 1000.0 * self.fps))
        if hi <= lo:
            return None

        seg = self.frames[lo:hi + n_q]
        score = np.zeros(hi - lo + 1, dtype=np.float32)
        for d in range(12):
            score += fftconvolve(seg[:, d], q[::-1, d], mode="valid")
        score /= n_q
        k = int(np.argmax(score))
        return Match(track_id=track_id,
                     offset_ms=int((lo + k - start + n_q) / self.fps * 1000),
                     confidence=float(score[k]), margin=0.0, rate=rate)

    def candidates(self, query: np.ndarray, top_k: int = 4,
                   rates=RATES) -> list[Match]:
        """Best few positions for the winning track, not just the best one.

        Repetition means the true position is often the second or third peak.
        Returning them all lets the caller carry every hypothesis forward and
        let time decide, instead of committing to a coin flip.
        """
        first = self.match(query, rates)
        if first is None:
            return []
        q = self._retime(query, first.rate)
        n_q = len(q)
        score = self._score(q)
        if score.size == 0:
            return [first]
        same = self.owner[:len(score)] == first.track_id
        score = np.where(same, score, -1.0)

        start = self.starts[first.track_id]
        out: list[Match] = []
        guard = int(6.0 * self.fps)      # keep peaks apart, not the same peak
        work = score.copy()
        for _ in range(top_k):
            k = int(np.argmax(work))
            if work[k] <= 0:
                break
            out.append(Match(
                track_id=first.track_id,
                offset_ms=int((k - start + n_q) / self.fps * 1000),
                confidence=float(work[k]), margin=first.margin, rate=first.rate))
            work[max(0, k - guard):k + guard] = -1.0
        return out or [first]

    def match(self, query: np.ndarray, rates=RATES) -> Match | None:
        """Identify a live window and locate it. `query` is (n, 12) chroma."""
        best = None
        for rate in rates:
            q = self._retime(query, rate)
            score = self._score(q)
            if score.size == 0:
                continue
            # A window must sit inside one track, never straddle two.
            valid = self.owner[:len(score)] == self.owner[len(q) - 1:len(q) - 1 + len(score)]
            score = np.where(valid, score, -1.0)
            k = int(np.argmax(score))
            if best is None or score[k] > best[0]:
                best = (float(score[k]), k, rate, score, len(q))

        if best is None:
            return None
        conf, k, rate, score, n_q = best
        tid = int(self.owner[k])

        # The margin that matters is over the best *other* track -- a second
        # peak inside the same record is the same answer, not a rival.
        rival = score.copy()
        rival[self.owner[:len(score)] == tid] = -1.0
        margin = conf - float(rival.max()) if rival.size else conf

        # Offset is reported at the end of the window: that is where the
        # needle is now, not where it was when the window opened.
        frames_in = k - self.starts[tid] + n_q
        return Match(track_id=tid, offset_ms=int(frames_in / self.fps * 1000),
                     confidence=conf, margin=max(0.0, margin), rate=rate)


if __name__ == "__main__":
    build()

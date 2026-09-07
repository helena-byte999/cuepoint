"""Build Cuepoint's database from a plain folder of audio files.

rekordbox supplies three things the scorer cannot read off a file on its own:
a tempo, a beat grid that knows where the bars start, and a phrase structure.
Depending on it means telling anyone who wants to try this tool to first buy
into a Pioneer ecosystem, import a collection and wait for it to analyse. That
is a large ask to make of someone who has not yet seen the thing work.

So this derives the same three things from the audio. What comes out goes into
exactly the tables `library.py` writes, which is the whole trick -- nothing
downstream can tell the difference. `extract` measures these tracks, `mix`
scores them, the UI plays them.

Where rekordbox is present it remains the better source: its grid is editable
by hand and its phrase model is trained, while everything here is measured
from one pass over the signal. `build` stays the default path; this is the
one for everybody else.

How good is it? Checked against rekordbox's own analysis of the same 60 files
-- afro house, gqom and amapiano, which is a fair test because that material
is where the offbeat is loudest:

    tempo agrees          55/60   (92%)
    grid on the beat      42/60   (70%)
    downbeat correct      26/42   (62% of the grids that were right)

So the tempo is reliable, the phase mostly is, and the downbeat is the weak
one. Read those three in order: each depends on the one above it, and an error
high up cannot be recovered further down.

  * Tempo is fitted as a single constant over the whole track. True of most
    machine-sequenced dance music, false of anything played by a band -- a
    live recording gets a grid that has drifted by the end. Every one of the
    five tempo disagreements above was a track rekordbox had also given a
    fractional tempo, which is its way of saying the same thing.
  * The downbeat is a heuristic: of the four candidate positions, the one
    carrying the most low end and the biggest spectral change. When it is
    wrong it is wrong by a whole beat, which rotates every groove template
    with it. Six other cues were tried against the collection above -- bar
    self-similarity, spectral novelty, low band and full band alone, and two
    combinations -- and none beat this one by more than a single track, so
    the honest summary is that it is unsolved rather than tuned.
  * Phrase roles are rules over energy and kick presence, not a trained
    segmenter. They separate "the drums have dropped out" from "this is the
    loudest the record gets", which is what the scorer's role term actually
    asks of them. They will not name a two-bar fill.
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio, features, library, paths

# Everything ffmpeg will decode and a DJ is likely to be holding.
AUDIO_EXT = {".mp3", ".m4a", ".mp4", ".wav", ".aif", ".aiff",
             ".flac", ".ogg", ".opus", ".wma"}

# The rhythm resolution features.py already uses -- 11.6 ms steps. Sharing it
# means the grid this fits and the grooves measured against it later come from
# the same view of the signal.
FPS = audio.SAMPLE_RATE / features.RHYTHM_HOP

BPM_MIN, BPM_MAX = 70.0, 180.0
# Autocorrelation cannot tell 64 from 128 from 256 -- they are the same
# pattern. A prior centred where dance music actually lives breaks the tie,
# and it is a soft one: 174 drum and bass still wins if the evidence is there.
BPM_PRIOR, PRIOR_WIDTH = 128.0, 0.30

# Coarse band split for describing what a bar sounds like. Boundaries follow
# how a mixer thinks -- sub, kick, low mid, and so on up.
BANDS = np.array([20., 60., 120., 250., 500., 1000., 2000., 4000., 8000., 11025.])

BLOCK_BARS = 4          # phrase boundaries land on 4-bar lines, never between
MIN_PHRASE_BLOCKS = 2   # so the shortest phrase is 8 bars
MIN_BARS = 16           # below this there is no structure worth segmenting

# Folder-scanned ids live above anything rekordbox will ever issue, so the two
# sources can share one database without a chance of collision.
_ID_BASE = 1 << 48


@dataclass
class Scan:
    """One track, described the way the database wants it."""
    tempo: float
    duration_ms: int
    beats_ms: np.ndarray          # every beat, in ms
    downbeat_offset: int          # index of the first beat that begins a bar
    phrases: list[tuple[str, int, int]]   # role, start_beat, end_beat (1-based)


def track_id(path: str | Path) -> int:
    """A stable id for a file, so re-scanning keeps the cached features.

    Derived from the path rather than handed out in sequence: `extract` caches
    to `features/<id>.npz`, and an id that shifted when a folder was rescanned
    would silently throw that work away.
    """
    h = hashlib.sha1(str(Path(path).resolve()).encode()).digest()
    return _ID_BASE + int.from_bytes(h[:6], "big")


# --------------------------------------------------------------------------
# tempo and grid
# --------------------------------------------------------------------------

def _autocorr(env: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation of the onset envelope, via FFT.

    np.correlate would be O(n^2) here, and n is around 31,000 frames for a
    six-minute record.
    """
    n = 1 << int(np.ceil(np.log2(2 * len(env))))
    f = np.fft.rfft(env - env.mean(), n)
    ac = np.fft.irfft(f * np.conj(f), n)[:len(env)]
    return ac / (ac[0] + 1e-12)


def estimate_tempo(env: np.ndarray, fps: float = FPS) -> float:
    """Best beat period in the envelope, in BPM."""
    ac = _autocorr(env)
    lags = np.arange(int(fps * 60 / BPM_MAX), int(fps * 60 / BPM_MIN) + 1)
    lags = lags[lags < len(ac)]
    if lags.size == 0:
        return BPM_PRIOR

    # A genuine beat period also repeats at the half-bar and the bar. Adding
    # those in rewards the lag that explains the whole hierarchy rather than
    # one that happens to fit a busy hi-hat.
    score = ac[lags].astype(np.float64).copy()
    for mult, weight in ((2, 0.5), (4, 0.25)):
        m = lags * mult
        ok = m < len(ac)
        score[ok] += weight * ac[m[ok]]

    bpm = 60 * fps / lags
    score *= np.exp(-0.5 * (np.log(bpm / BPM_PRIOR) / PRIOR_WIDTH) ** 2)
    return float(bpm[int(np.argmax(score))])


def _fold(env: np.ndarray, period: float, start: int = 0,
          nbins: int = 64) -> np.ndarray:
    """Sum the envelope into `nbins` bins of one beat period.

    Folding the whole track onto a single beat is the cheap way to ask how
    well a period explains it: the right one piles every onset into the same
    bin, a wrong one smears them flat. `start` is the index of `env[0]` within
    the full envelope, so a window folded here reports its phase against the
    beginning of the track rather than the beginning of the window.
    """
    pos = np.mod(np.arange(start, start + len(env), dtype=np.float64), period)
    b = np.minimum((pos / period * nbins).astype(np.int64), nbins - 1)
    return np.bincount(b, weights=env, minlength=nbins)


def _contrast(hist: np.ndarray) -> float:
    """How peaked a folded envelope is. Every fold holds the same total
    energy, so this is really asking how much of it landed in one bin."""
    return float(hist.max() / (hist.mean() + 1e-12))


# Fractions of the track to fold, and how wide to search around the running
# estimate. Coarse to fine, and the two move together for a reason: the
# objective sharpens as the window grows. Over 500 beats a tempo wrong by a
# tenth of a percent has drifted half a beat by the end and folds flat, so
# the full-length pass can only ever refine an estimate that is already close
# -- searched at that resolution from the start it would step straight over
# the answer. A short window has a peak broad enough to find, and a long one
# has the precision to sharpen it.
# (seconds of audio to fold, half-width of the search, steps). None means the
# whole track. The first window is a fixed length rather than a fraction of
# the record on purpose: how finely this has to search depends on how many
# beats are in the window, so a ten-minute mix would otherwise get a first
# pass sharper than its own step size and skip straight over the answer.
_STAGES = ((60.0, 0.06, 201), (None, 0.005, 321))


def fit_grid(env: np.ndarray, bpm0: float, fps: float = FPS,
             phase_env: np.ndarray | None = None) -> tuple[float, float]:
    """Refine the tempo and find the phase. Returns (bpm, first beat in frames).

    A beat's worth of error accumulated over six minutes is several bars out
    by the end, so this is worth doing properly.

    The period is measured from everything that happens, because more onsets
    make the periodicity clearer. The phase is better measured from `phase_env`
    -- the low band alone -- because the two questions have different answers:
    how often the record repeats is settled by the whole mix, but *where* the
    beat falls is settled by the kick. In a lot of gqom and amapiano the
    loudest thing in the bar is an offbeat shaker, and a grid placed on the
    loudest onset lands squarely between the beats.
    """
    n = len(env)
    bpm, phase = bpm0, 0.0
    for seconds, span, steps in _STAGES:
        if seconds is None:
            seg, start = env, 0
        else:
            # The middle of the record: an intro can be too sparse to fold.
            width = min(n, max(int(seconds * fps), 1))
            start = (n - width) // 2
            seg = env[start:start + width]
        best = -1.0
        for cand in np.linspace(bpm * (1 - span), bpm * (1 + span), steps):
            period = 60 * fps / cand
            hist = _fold(seg, period, start)
            score = _contrast(hist)
            if score > best:
                best = score
                bpm_next = float(cand)
                phase_next = (float(np.argmax(hist)) + 0.5) / len(hist) * period
        bpm, phase = bpm_next, phase_next

    period = 60 * fps / bpm
    if phase_env is not None:
        low = _fold(phase_env, period)
        # Only when the low band actually has a pulse to offer. An acapella
        # has no kick to find, and its fold is flat noise -- there the whole
        # mix is the only evidence there is.
        if _contrast(low) > 1.3:
            phase = (float(np.argmax(low)) + 0.5) / len(low) * period
    return bpm, math.fmod(phase, period)


def beat_frames(n_frames: int, bpm: float, phase: float,
                fps: float = FPS) -> np.ndarray:
    """Every beat position, in (fractional) frame units."""
    period = 60 * fps / bpm
    count = int(max(0, (n_frames - 1 - phase)) // period) + 1
    return phase + np.arange(count) * period


# --------------------------------------------------------------------------
# what each beat sounds like
# --------------------------------------------------------------------------

def beat_bands(spec: np.ndarray, freqs: np.ndarray,
               beats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """What each beat sounds like: (log band shape, linear band energy).

    Two views of the same measurement, because two different questions get
    asked of it. Segmentation compares timbre, and wants the log -- a hi-hat
    entering matters as much as a bass note, though it carries a thousandth
    of the energy. The role rules ask how loud the record is, and want the
    linear energy, or a broadband riser reads louder than the drop it is
    building to.

    Shape is (beats - 1, bands) for both.
    """
    masks = [(freqs >= BANDS[i]) & (freqs < BANDS[i + 1])
             for i in range(len(BANDS) - 1)]
    per_band = np.stack([spec[m].sum(axis=0) for m in masks])   # (bands, frames)

    lin = np.zeros((max(0, len(beats) - 1), per_band.shape[0]))
    for i, (a, b) in enumerate(zip(beats[:-1], beats[1:])):
        lo = max(0, int(round(a)))
        hi = min(per_band.shape[1], max(lo + 1, int(round(b))))
        if lo >= per_band.shape[1]:
            break
        lin[i] = per_band[:, lo:hi].mean(axis=1)
    return np.log1p(1000.0 * lin), lin


def _db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(x + 1e-9)


def _normalise(x: np.ndarray) -> np.ndarray:
    """Rescale to 0..1 against the track's own 10th and 90th percentiles.

    Against its own, not an absolute scale: what matters is whether this is a
    loud part of *this* record, and mastering levels differ by more between
    two records than the sections do within one.
    """
    lo = float(np.percentile(x, 10))
    hi = float(np.percentile(x, 90))
    return np.clip((x - lo) / max(hi - lo, 1e-9), -0.5, 1.5)


def _novelty(rows: np.ndarray) -> np.ndarray:
    """How different each row is from the one before it.

    Each band is standardised across the track first, and that step is doing
    the real work. Raw band energies are overwhelmingly a constant -- every
    bar of a record is mostly the same shape -- so the angle between two
    neighbours is a fraction of a degree, and the interesting variation sits
    in the last few digits of a number close to one. Centring each band on
    its own mean throws the constant away and leaves only what changed, which
    is the thing being measured.
    """
    if len(rows) < 2:
        return np.zeros(len(rows))
    z = rows - rows.mean(axis=0)
    z /= (z.std(axis=0) + 1e-12)
    norm = np.linalg.norm(z, axis=1) + 1e-12
    unit = z / norm[:, None]
    sim = np.sum(unit[1:] * unit[:-1], axis=1)
    return np.concatenate([[0.0], 1.0 - sim])


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def find_downbeat(bands: np.ndarray, low_at_beat: np.ndarray) -> int:
    """Which of the four beat positions starts the bar.

    Two cues, both weak alone and reasonable together: a bar tends to open on
    the heaviest low end, and it tends to be where the arrangement changes --
    a new element arrives on the one, not on the three.
    """
    n = min(len(bands), len(low_at_beat))
    if n < 8:
        return 0
    weight = _z(low_at_beat[:n])
    change = _z(_novelty(bands[:n]))
    scores = [float(weight[o::4].mean() + 0.5 * change[o::4].mean())
              for o in range(4)]
    return int(np.argmax(scores))


# --------------------------------------------------------------------------
# phrases
# --------------------------------------------------------------------------

def segment(block_feat: np.ndarray) -> list[int]:
    """Block indices where a new phrase starts, always including 0.

    Dance music changes on 4-bar lines, so the candidates are the lines
    themselves and the only question is which of them are real. A boundary
    qualifies when the sound either side of it differs more than this track's
    usual drift from one block to the next, and when it is a local maximum -- otherwise a
    long ramp registers as a boundary at every step along it.
    """
    n = len(block_feat)
    if n < 2 * MIN_PHRASE_BLOCKS:
        return [0]

    change = _novelty(block_feat)
    inner = change[1:]
    # Median absolute deviation, not standard deviation: a single enormous
    # change (the drop) would inflate sigma enough to hide every other one.
    med = float(np.median(inner))
    mad = float(np.median(np.abs(inner - med))) or float(inner.std()) or 1e-9
    threshold = med + 1.6 * mad

    bounds = [0]
    for i in range(1, n):
        if change[i] < threshold:
            continue
        if i > 1 and change[i] <= change[i - 1]:
            continue
        if i + 1 < n and change[i] < change[i + 1]:
            continue
        if i - bounds[-1] < MIN_PHRASE_BLOCKS or n - i < MIN_PHRASE_BLOCKS:
            continue
        bounds.append(i)
    return bounds


def label_roles(spans: list[tuple[int, int]], bar_energy: np.ndarray,
                bar_kick: np.ndarray) -> list[str]:
    """Name each phrase, in the vocabulary the scorer's role term expects.

    rekordbox has a trained model for this. These are rules, and they lean on
    the two things that are unambiguous in the signal: whether the kick is
    playing, and how loud the record is against its own peak. That is enough
    to separate the cases the scorer actually distinguishes -- a good place to
    leave from, a good place to arrive at, and the two extremes.

    Both inputs are per-bar and already scaled 0..1 across the track.
    """
    if not spans:
        return []
    level, rise, kick = [], [], []
    for a, b in spans:
        e = bar_energy[a:b]
        quarter = max(1, len(e) // 4)
        level.append(float(e.mean()))
        rise.append(float(e[-quarter:].mean() - e[:quarter].mean()))
        kick.append(float(bar_kick[a:b].mean()))

    roles = []
    for n, (a, _) in enumerate(spans):
        prev = level[n - 1] if n else 0.0
        nxt = level[n + 1] if n + 1 < len(spans) else 0.0
        # A build is a phrase on its way somewhere: either it climbs while you
        # are in it, or it sits between something quieter and something louder.
        # The second test matters because a long build often gets segmented in
        # two, and neither half climbs much on its own.
        climbing = rise[n] > 0.15 or (level[n] > prev and level[n] < nxt - 0.12)

        if n == 0:
            roles.append("intro")
        elif n == len(spans) - 1:
            roles.append("outro")
        elif kick[n] < 0.35:
            # The drums are out. Early on that is still the intro; anywhere
            # else it is the breakdown, and both are good places to arrive.
            roles.append("intro" if a < 0.12 * len(bar_energy) else "breakdown")
        elif climbing:
            roles.append("build")
        elif level[n] > 0.80:
            roles.append("peak")
        else:
            roles.append("verse")
    return roles


# --------------------------------------------------------------------------
# one track, end to end
# --------------------------------------------------------------------------

def analyse(path: str | Path) -> Scan:
    """Tempo, beat grid and phrase structure for one audio file."""
    y = audio.decode(path)
    if y.size < 30 * audio.SAMPLE_RATE:
        raise audio.DecodeError(f"{Path(path).name}: under 30s, nothing to grid")
    duration_ms = int(1000 * len(y) / audio.SAMPLE_RATE)

    spec = audio.spectrogram(y, features.RHYTHM_NFFT, features.RHYTHM_HOP)
    freqs = audio.fft_freqs(features.RHYTHM_NFFT)
    env_full = audio.onset_envelope(spec, freqs)
    env_low = audio.onset_envelope(spec, freqs, features.LOW_BAND)

    bpm, phase = fit_grid(env_full, estimate_tempo(env_full), phase_env=env_low)
    beats = beat_frames(spec.shape[1], bpm, phase)
    if len(beats) < 4 * MIN_BARS:
        raise audio.DecodeError(f"{Path(path).name}: too few beats to segment")

    # frame_times centres each window, so the grid lands where the onsets
    # actually are rather than half a window early.
    times = audio.frame_times(spec.shape[1], features.RHYTHM_HOP,
                              features.RHYTHM_NFFT)
    beats_ms = np.interp(beats, np.arange(len(times)), times) * 1000.0

    bands, lin = beat_bands(spec, freqs, beats)
    low_at_beat = np.interp(beats[:len(bands)], np.arange(len(env_low)), env_low)
    offset = find_downbeat(bands, low_at_beat)

    # Bars, from the first complete one. Anything before the downbeat is a
    # pickup and belongs to no bar.
    n_bars = (len(bands) - offset) // 4
    if n_bars < MIN_BARS:
        raise audio.DecodeError(f"{Path(path).name}: too short to segment")
    bar_feat = bands[offset:offset + 4 * n_bars].reshape(n_bars, 4, -1).mean(axis=1)
    bar_lin = lin[offset:offset + 4 * n_bars].reshape(n_bars, 4, -1).mean(axis=1)

    # Loudness, and the low end on its own -- the two the role rules read.
    bar_energy = _normalise(_db(bar_lin.sum(axis=1)))
    bar_kick = _normalise(_db(bar_lin[:, :2].sum(axis=1)))   # sub and kick bands

    n_blocks = n_bars // BLOCK_BARS
    block_feat = bar_feat[:n_blocks * BLOCK_BARS].reshape(
        n_blocks, BLOCK_BARS, -1).mean(axis=1)
    bounds = segment(block_feat)

    spans_bars = []
    for i, start in enumerate(bounds):
        if i + 1 < len(bounds):
            end = bounds[i + 1] * BLOCK_BARS
        else:
            end = n_bars          # the tail is short of a whole block; the
        spans_bars.append((start * BLOCK_BARS, end))   # last phrase keeps it
    roles = label_roles(spans_bars, bar_energy, bar_kick)

    phrases = []
    for (a, b), role in zip(spans_bars, roles):
        # Beat indices are 1-based, and bar `a` starts `offset` beats in.
        start_beat = offset + a * 4 + 1
        end_beat = offset + b * 4 + 1
        phrases.append((role, start_beat, min(end_beat, len(beats))))

    return Scan(tempo=bpm, duration_ms=duration_ms,
                beats_ms=beats_ms, downbeat_offset=offset, phrases=phrases)


# --------------------------------------------------------------------------
# a folder of records -> the database
# --------------------------------------------------------------------------

def find_files(roots: list[str | Path]) -> list[Path]:
    """Every audio file under the given folders, sorted, deduplicated."""
    seen: dict[str, Path] = {}
    for root in roots:
        r = Path(root).expanduser()
        if r.is_file():
            if r.suffix.lower() in AUDIO_EXT:
                seen[str(r.resolve())] = r
            continue
        for f in sorted(r.rglob("*")):
            # Skip the resource forks and hidden junk macOS scatters through
            # a music folder; ffmpeg would fail on all of them.
            if f.name.startswith("._") or f.name.startswith("."):
                continue
            if f.is_file() and f.suffix.lower() in AUDIO_EXT:
                seen[str(f.resolve())] = f
    return [seen[k] for k in sorted(seen)]


def _drop_cache(tid: int) -> None:
    """Forget a track's measured features, so `extract` takes them again."""
    try:
        (paths.features_dir() / f"{tid}.npz").unlink()
    except OSError:
        pass


def _job(path_str: str):
    try:
        return path_str, analyse(path_str), ""
    except Exception as exc:                             # noqa: BLE001
        return path_str, None, f"{type(exc).__name__}: {exc}"[:90]


def signature(con: sqlite3.Connection, tid: int) -> tuple:
    """What a track's cached features were measured against.

    `extract` decides whether to re-measure by comparing file dates, which is
    the right test when only the audio can change. Here the *analysis* can
    change too -- a rescan can move a phrase boundary with the file untouched
    -- and features measured over the old phrases would then be handed to the
    scorer as though they described the new ones. So the grid and the phrases
    are compared directly, and a track whose structure moved loses its cache.
    """
    row = con.execute("SELECT tempo FROM tracks WHERE id=?", (tid,)).fetchone()
    if row is None:
        return ()
    phrases = con.execute(
        "SELECT idx, role, start_beat, end_beat FROM phrases "
        "WHERE track_id=? ORDER BY idx", (tid,)).fetchall()
    return (round(row[0] or 0.0, 3), tuple(map(tuple, phrases)))


def _write(con: sqlite3.Connection, path: Path, s: Scan) -> bool:
    """Write one scanned track. True if its structure changed."""
    tid = track_id(path)
    before = signature(con, tid)
    con.execute(
        "INSERT OR REPLACE INTO tracks "
        "(id, path, filename, title, folder, present, duration_ms, tempo, "
        " beat_count, mood, analyzed, source) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (tid, str(path), path.name, library._clean_title(path.name),
         path.parent.name, 1, s.duration_ms, s.tempo, len(s.beats_ms),
         1, 1, "folder"),
    )
    for table in ("beats", "phrases", "cues"):
        con.execute(f"DELETE FROM {table} WHERE track_id=?", (tid,))

    con.executemany(
        "INSERT INTO beats VALUES (?,?,?,?,?)",
        [(tid, i, (i - s.downbeat_offset) % 4 + 1, s.tempo, int(round(ms)))
         for i, ms in enumerate(s.beats_ms)],
    )

    def ms_of(beat: int) -> int:
        i = max(0, min(beat - 1, len(s.beats_ms) - 1))
        return int(round(s.beats_ms[i]))

    con.executemany(
        "INSERT INTO phrases VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(tid, i, role, 0, start, end,
          (start - 1 - s.downbeat_offset) // 4 + 1,
          max(1, (end - start) // 4), ms_of(start), ms_of(end))
         for i, (role, start, end) in enumerate(s.phrases)],
    )
    return signature(con, tid) != before


def build(roots: list[str | Path], db_path: str | Path | None = None,
          workers: int | None = None, verbose: bool = True) -> dict:
    """Scan folders of audio and write them into Cuepoint's database.

    Additive by folder: scanning a second crate adds to the first, and
    re-scanning one refreshes only that one. Tracks that have gone from a
    scanned folder go with it, so a rescan is also how you clean up.
    """
    paths.ensure()
    files = find_files(roots)
    if not files:
        if verbose:
            print("No audio files found in: " + ", ".join(str(r) for r in roots))
        return {"found": 0, "ok": 0, "failed": 0}

    con = sqlite3.connect(db_path or paths.db_path())
    con.executescript(library.SCHEMA)
    library.migrate(con)

    # Tracks previously scanned from under these roots whose files have since
    # gone. Only those: everything still present is rewritten in place below,
    # so that its cached features survive a rescan that did not change it.
    prefixes = [str(Path(r).expanduser().resolve()) for r in roots]
    keep = {track_id(f) for f in files}
    gone = [row[0] for row in con.execute(
        "SELECT id, path FROM tracks WHERE source='folder'")
        if row[0] not in keep
        and any(row[1] == p or row[1].startswith(p + os.sep) for p in prefixes)]
    for tid in gone:
        for table in ("tracks", "beats", "phrases", "cues"):
            col = "id" if table == "tracks" else "track_id"
            con.execute(f"DELETE FROM {table} WHERE {col}=?", (tid,))
        _drop_cache(tid)

    workers = workers or max(1, min(len(files), (os.cpu_count() or 4) - 1))
    if verbose:
        print(f"Scanning {len(files)} files across {workers} workers...")

    ok, changed, failures, t0 = 0, 0, [], time.time()
    with mp.Pool(workers) as pool:
        for i, (path_str, s, err) in enumerate(
                pool.imap_unordered(_job, [str(f) for f in files], chunksize=1), 1):
            if s is not None:
                if _write(con, Path(path_str), s):
                    _drop_cache(track_id(path_str))
                    changed += 1
                ok += 1
            else:
                failures.append((Path(path_str).name, err))
            if verbose:
                done = time.time() - t0
                print(f"\r  {i}/{len(files)}  ok={ok}  failed={len(failures)}  "
                      f"eta {done / i * (len(files) - i):>4.0f}s",
                      end="", flush=True)

    con.commit()
    con.close()

    if verbose:
        print(f"\nScanned in {time.time() - t0:.0f}s -- {ok} usable, "
              f"{len(failures)} skipped, {changed} new or changed.")
        for name, why in failures[:10]:
            print(f"  {name}: {why}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")
    return {"found": len(files), "ok": ok, "changed": changed,
            "failed": len(failures)}

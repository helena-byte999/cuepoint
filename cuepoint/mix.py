"""Deciding what to play next, and exactly where to blend it.

This is the part that earns the rest of the project. Everything upstream --
the ANLZ parser, the beat grid, the per-phrase features -- exists so that this
module can answer one question: *given that I am playing this record, which
other record in the crate will sit on top of it, and at which bar?*

A recommendation is therefore never just a track. It is a `Blend`: a phrase of
the outgoing track, a phrase of the incoming track, and the number of bars the
two are expected to run together.

How a blend is judged
---------------------
Seven components, each scored 0..1 and combined by weight:

    tempo     can the pitch fader even get there (also half/double time)
    groove    do the two kick patterns land on the same slots of the bar
    vocal     are two people singing at once
    harmonic  do the pitch classes agree, weighted by how tonal each side is
    energy    does the incoming record hold or lift the level
    role      is this a musically sensible place to leave and to arrive
    sub       do the two kick fundamentals fight in the bottom octave

Tempo is also a hard gate: past the pitch range there is no blend to score.

The thresholds below are not guesses. They were measured across every phrase
in this collection -- e.g. two phrases picked at random correlate at 0.15 in
the low band, so a correlation of 0.6 is a real rhythmic agreement and not the
baseline. The measured percentiles are quoted next to each constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import features as featmod, library, paths

CACHE = None    # resolved per call via paths.features_dir()

# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
# Percentiles quoted are over all 3,000+ analysed phrases in the collection.

MAX_STRETCH = 0.08        # pitch fader range, +/- 8%
DOUBLE_TIME_PENALTY = 0.9  # half/double-time blends work, but are a harder move

# groove correlation between random phrases: p50 0.14, p90 0.58
GROOVE_BAD, GROOVE_GOOD = -0.20, 0.70

# The low band carries the clash risk -- two kicks on different sixteenths is
# the wreck everybody hears -- so it leads. The full band is not redundant with
# it: the two similarity measures agree at only r=0.38 across the collection,
# so 85% of what the full band says about a pair is information the low band
# does not have. It describes the hats and percussion, where a swung record
# meeting a straight one gives itself away.
GROOVE_LOW_SHARE = 0.7

# vocal: p25 0.20, p50 0.24, p90 0.35
VOCAL_QUIET, VOCAL_LOUD = 0.20, 0.36

# chroma correlation: random phrase pairs sit at p50 0.13, while two phrases
# of the *same* record -- guaranteed same key -- sit at p50 0.81. That gap is
# the signal, and the ramp is set across it.
HARMONIC_BAD, HARMONIC_GOOD = -0.10, 0.70

ENERGY_IDEAL = 1.0        # dB: a touch of lift is the ideal landing
ENERGY_RISE, ENERGY_DROP = 5.0, 4.0   # dB tolerance either side of ideal

WEIGHTS = {
    "tempo": 0.15,
    "groove": 0.22,
    "vocal": 0.18,
    "harmonic": 0.15,
    "energy": 0.12,
    "role": 0.13,
    "sub": 0.05,
}

# How good each role is to leave from, and to arrive on. A blend's structural
# score is the geometric mean of the two, so a bad half can't be averaged away
# by a good one.
_EXIT = {"outro": 1.00, "breakdown": 0.90, "bridge": 0.70, "verse": 0.70,
         "build": 0.55, "peak": 0.45, "intro": 0.20}
_ENTRY = {"intro": 1.00, "build": 0.85, "verse": 0.70, "breakdown": 0.60,
          "bridge": 0.60, "peak": 0.50, "outro": 0.15}

# Pairs that the affinities above rate too kindly. Two peaks at once is the
# double drop -- it exists, but it is a stunt, not a transition; two
# breakdowns in a row simply stops the night.
_PAIR_ADJUST = {("peak", "peak"): 0.70, ("breakdown", "breakdown"): 0.70}

MIX_BARS = (32, 16, 8, 4)   # the lengths a DJ actually counts
MIN_PHRASE_BARS = 4         # shorter than this is a fill, not a mix point


def _ramp(x: float, lo: float, hi: float) -> float:
    """Linear 0..1 ramp, clamped outside [lo, hi]."""
    if hi == lo:
        return 1.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two vectors.

    Correlation rather than cosine: both groove and chroma vectors sit well
    above zero everywhere, so cosine similarity is high for any pair and
    discriminates poorly. Removing the mean asks the question we actually
    care about -- do the peaks line up.
    """
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------

@dataclass
class Phrase:
    idx: int
    role: str
    start_ms: int
    end_ms: int
    start_bar: int
    bars: int
    groove_low: np.ndarray
    groove_full: np.ndarray
    chroma: np.ndarray
    kick_hz: float
    energy_db: float
    vocal: float

    @property
    def tonality(self) -> float:
        """How pitched this phrase is, 0 (drums only) .. 1 (clear harmony)."""
        c = self.chroma
        if c.size == 0 or c.max() <= 0:
            return 0.0
        return float((c.max() - c.mean()) / c.max())


@dataclass
class Track:
    id: int
    title: str
    filename: str
    tempo: float
    duration_ms: int
    # The folder a track sits in is the only genre signal this collection
    # carries -- the files themselves arrive from download sites with their
    # tags stripped, and rekordbox keeps genre in the encrypted master.db.
    folder: str = ""
    path: str = ""            # so the UI can play the record it is scoring
    phrases: list[Phrase] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        """Track key, from the chroma of its most tonal phrases."""
        if not self.phrases:
            return "?", "?"
        weights = np.array([p.tonality for p in self.phrases])
        stack = np.stack([p.chroma for p in self.phrases])
        if weights.sum() <= 0:
            return "?", "?"
        return featmod.estimate_key((stack * weights[:, None]).sum(axis=0))


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_track(con, track_id: int, cache: Path | None = None) -> Track | None:
    """Assemble one track from the database plus its cached features.

    Returns None when the track has no features on disk -- it cannot take
    part in a blend, so there is nothing to build.
    """
    row = con.execute(
        "SELECT id, title, filename, tempo, duration_ms, folder, path "
        "FROM tracks WHERE id=?", (track_id,)).fetchone()
    if row is None:
        return None
    cache = cache or paths.features_dir()
    path = cache / f"{track_id}.npz"
    if not path.exists():
        return None
    f = featmod.load(path)
    by_idx = {int(v): i for i, v in enumerate(f["idx"])}

    track = Track(id=row["id"], title=row["title"], filename=row["filename"],
                  tempo=row["tempo"] or 0.0, duration_ms=row["duration_ms"] or 0,
                  folder=row["folder"] or "", path=row["path"] or "")
    for p in con.execute(
            "SELECT idx, role, start_ms, end_ms, start_bar, bars FROM phrases "
            "WHERE track_id=? ORDER BY idx", (track_id,)):
        i = by_idx.get(p["idx"])
        if i is None:          # phrase was skipped during extraction
            continue
        track.phrases.append(Phrase(
            idx=p["idx"], role=p["role"], start_ms=p["start_ms"],
            end_ms=p["end_ms"], start_bar=p["start_bar"], bars=p["bars"],
            groove_low=f["groove_low"][i], groove_full=f["groove_full"][i],
            chroma=f["chroma"][i], kick_hz=float(f["kick_hz"][i]),
            energy_db=float(f["energy_db"][i]), vocal=float(f["vocal"][i]),
        ))
    return track if track.phrases else None


def load_all(con, cache: Path | None = None) -> dict[int, Track]:
    """Every track that is ready to be mixed, keyed by id."""
    cache = cache or paths.features_dir()
    ids = [r["id"] for r in con.execute(
        "SELECT id FROM tracks WHERE analyzed=1 AND present=1 ORDER BY id")]
    out = {}
    for tid in ids:
        t = load_track(con, tid, cache)
        if t is not None:
            out[tid] = t
    return out


# --------------------------------------------------------------------------
# the seven components
# --------------------------------------------------------------------------

def tempo_fit(a_bpm: float, b_bpm: float) -> tuple[float, float, float]:
    """Score the tempo gap. Returns (score, ratio, stretch).

    `ratio` is what B's tempo must be multiplied by to meet A: 1 for a normal
    blend, 2 when B is a half-time record running underneath, 0.5 when it is
    double-time on top. `stretch` is the pitch-fader move as a fraction, so
    0.03 means "+3% on the incoming deck".
    """
    if a_bpm <= 0 or b_bpm <= 0:
        return 0.0, 1.0, 0.0
    best = (0.0, 1.0, 0.0)
    for ratio in (1.0, 2.0, 0.5):
        stretch = (b_bpm * ratio) / a_bpm - 1.0
        if abs(stretch) > MAX_STRETCH:
            continue
        score = 1.0 - abs(stretch) / MAX_STRETCH
        if ratio != 1.0:
            score *= DOUBLE_TIME_PENALTY
        if score > best[0]:
            best = (score, ratio, stretch)
    return best


def _rescale(v: np.ndarray, n: int) -> np.ndarray:
    """Resample a bar-relative vector onto `n` slots, wrapping at the bar."""
    src = np.arange(len(v) + 1, dtype=float) / len(v)
    dst = (np.arange(n) + 0.5) / n
    return np.interp(dst, src, np.append(v, v[0]))


def _groove_views(g: np.ndarray, ratio: float) -> list[np.ndarray]:
    """B's one-bar groove, expressed on A's bar.

    At 1:1 the two bars are the same length and the vectors compare directly.
    Under half/double time they are not, and comparing them raw would be
    meaningless -- so the incoming groove is stretched or tiled onto A's bar
    first. A half-time record's bar spans two of A's, and either half of it
    could be the one that lands, so both are offered and the better wins.
    """
    n = len(g)
    if ratio == 1.0:
        return [g]
    if ratio == 2.0:          # B is half-time: its bar covers two of A's
        half = n // 2
        return [_rescale(g[:half], n), _rescale(g[half:], n)]
    # ratio 0.5 -- B is double-time: its bar fits twice inside A's
    return [np.concatenate([_rescale(g, n // 2), _rescale(g, n // 2)])]


def _band_fit(ga: np.ndarray, gb: np.ndarray, ratio: float) -> float:
    """Best correlation between two bar-relative grooves, over B's alignments."""
    return _ramp(max(_corr(ga, view) for view in _groove_views(gb, ratio)),
                 GROOVE_BAD, GROOVE_GOOD)


def groove_fit(a: Phrase, b: Phrase, ratio: float = 1.0) -> float:
    """Do the two records agree about where the weight of the bar falls?

    This is the component that catches the train wreck a key-and-BPM tool
    cannot see: two records at the same tempo and in the same key whose kicks
    land on different sixteenths still fight each other. Scored in two bands --
    the low one for that clash, the full one for whether the overall feel
    matches -- and mixed by GROOVE_LOW_SHARE.
    """
    low = _band_fit(a.groove_low, b.groove_low, ratio)
    full = _band_fit(a.groove_full, b.groove_full, ratio)
    return GROOVE_LOW_SHARE * low + (1.0 - GROOVE_LOW_SHARE) * full


def vocal_fit(a: Phrase, b: Phrase) -> float:
    """Penalise two vocals at once -- the most audible mistake in the list.

    The penalty is the product of both sides, so an instrumental passage on
    either deck clears it completely. That is exactly the real rule: you can
    bring a vocal in over a beat, or a beat in under a vocal, but not both.
    """
    va = _ramp(a.vocal, VOCAL_QUIET, VOCAL_LOUD)
    vb = _ramp(b.vocal, VOCAL_QUIET, VOCAL_LOUD)
    return 1.0 - va * vb


def harmonic_fit(a: Phrase, b: Phrase) -> float:
    """Do the two phrases agree about pitch?

    Straight correlation of the 12-vectors. The obvious refinement -- discount
    the score for drum-only phrases, which have no key to clash with -- was
    tried and dropped: the available proxy for "how tonal is this" (peakedness
    of the chroma vector) turned out not to separate tonal from percussive
    material at all. Pairs that measured as tonal on both sides spread no
    differently from pairs that measured as flat, so the discount only
    flattened a signal that is otherwise strong. Left out until there is a
    tonality measure that earns its place.
    """
    return _ramp(_corr(a.chroma, b.chroma), HARMONIC_BAD, HARMONIC_GOOD)


def energy_fit(a: Phrase, b: Phrase) -> float:
    """Reward a level that holds or lifts slightly; punish the floor dying.

    Asymmetric on purpose: coming in 5 dB hot is a choice, coming in 5 dB
    quiet is a mistake.
    """
    d = b.energy_db - a.energy_db - ENERGY_IDEAL
    sigma = ENERGY_RISE if d >= 0 else ENERGY_DROP
    return float(math.exp(-(d / sigma) ** 2))


def role_fit(a: Phrase, b: Phrase) -> float:
    """Is this a sensible place to leave, and a sensible place to arrive?"""
    exit_, entry = _EXIT.get(a.role, 0.5), _ENTRY.get(b.role, 0.5)
    score = math.sqrt(exit_ * entry)
    return score * _PAIR_ADJUST.get((a.role, b.role), 1.0)

def sub_fit(a: Phrase, b: Phrase) -> float:
    """Do the two kick fundamentals share the bottom octave peacefully?

    Two kicks tuned to the same note lock together. Tuned a semitone or two
    apart they beat against each other and the low end turns to mud. Far
    apart they simply read as two different drums, which is fine. A mild
    heuristic on a low weight -- it breaks ties, it does not decide them.
    """
    if a.kick_hz <= 0 or b.kick_hz <= 0:
        return 0.85
    semitones = abs(12.0 * math.log2(b.kick_hz / a.kick_hz))
    if semitones < 0.35:
        return 1.0            # same note: they reinforce
    if semitones < 3.0:
        return 0.55           # close enough to beat against each other
    return 0.85               # clearly separate registers


# --------------------------------------------------------------------------
# blends
# --------------------------------------------------------------------------

@dataclass
class Blend:
    track: Track
    from_phrase: Phrase
    to_phrase: Phrase
    score: float
    parts: dict[str, float]
    ratio: float          # 1.0, or 2.0/0.5 for a half/double-time blend
    stretch: float        # pitch-fader move on the incoming deck, as a fraction
    bars: int             # how long to run the two together

    @property
    def cue_out_ms(self) -> int:
        """Where in the outgoing track the blend begins."""
        return self.from_phrase.start_ms

    @property
    def cue_in_ms(self) -> int:
        """Where in the incoming track to set the play head."""
        return self.to_phrase.start_ms

    @property
    def bpm(self) -> float:
        """The tempo the incoming deck ends up at, once pitched to match."""
        return self.track.tempo * self.ratio * (1.0 + self.stretch)

    def weakest(self) -> tuple[str, float]:
        """The component costing this blend the most score.

        Weighted by how much each component actually matters, otherwise the
        answer is always `sub` -- a deliberate tie-breaker at 5% weight that
        would otherwise shout louder than a genuine vocal clash.
        """
        return min(self.parts.items(),
                   key=lambda kv: (1.0 - kv[1]) * -WEIGHTS.get(kv[0], 0.0))


def _mix_bars(a: Phrase, b: Phrase, ratio: float) -> int:
    """How many bars to run the two records together.

    Bounded by both phrases -- there is no point counting 32 bars over an
    8-bar intro -- and rounded down to a length a DJ would actually count.
    Under half/double time the incoming track's bars are a different length,
    so they are converted to the outgoing track's bars first.
    """
    limit = min(a.bars, b.bars * ratio)
    for n in MIX_BARS:
        if n <= limit:
            return n
    return MIX_BARS[-1]


def score_blend(a_track: Track, a: Phrase, b_track: Track, b: Phrase,
                weights: dict[str, float] | None = None) -> Blend | None:
    """Score one specific transition. None when the tempos cannot meet."""
    w = weights or WEIGHTS
    t_score, ratio, stretch = tempo_fit(a_track.tempo, b_track.tempo)
    if t_score <= 0.0:
        return None            # outside pitch range: not a blend at all

    parts = {
        "tempo": t_score,
        "groove": groove_fit(a, b, ratio),
        "vocal": vocal_fit(a, b),
        "harmonic": harmonic_fit(a, b),
        "energy": energy_fit(a, b),
        "role": role_fit(a, b),
        "sub": sub_fit(a, b),
    }
    total = sum(parts[k] * w[k] for k in parts) / sum(w.values())
    return Blend(track=b_track, from_phrase=a, to_phrase=b, score=total,
                 parts=parts, ratio=ratio, stretch=stretch,
                 bars=_mix_bars(a, b, ratio))


def exit_phrases(track: Track, limit: int = 4) -> list[Phrase]:
    """Phrases of the outgoing track worth mixing out of.

    A DJ leaves late, so candidates are drawn from the back half, ranked by
    how good their role is to leave from. The final phrase is always included
    -- if nothing else works, you run the record to its end.
    """
    if not track.phrases:
        return []
    end = track.phrases[-1].end_ms or track.duration_ms
    late = [p for p in track.phrases
            if p.start_ms >= 0.5 * end and p.bars >= MIN_PHRASE_BARS]
    late = late or [p for p in track.phrases if p.bars >= MIN_PHRASE_BARS][-3:]
    late = late or track.phrases[-3:]
    ranked = sorted(late, key=lambda p: -_EXIT.get(p.role, 0.5))[:limit]
    if track.phrases[-1] not in ranked:
        ranked.append(track.phrases[-1])
    return ranked


def entry_phrases(track: Track, limit: int = 4) -> list[Phrase]:
    """Phrases of an incoming track worth arriving on.

    The front half, ranked by how good the role is to come in on -- plus the
    very first phrase, which is what you get if you simply press play.
    """
    if not track.phrases:
        return []
    end = track.phrases[-1].end_ms or track.duration_ms
    early = [p for p in track.phrases
             if p.start_ms <= 0.5 * end and p.bars >= MIN_PHRASE_BARS]
    early = early or [p for p in track.phrases if p.bars >= MIN_PHRASE_BARS][:3]
    early = early or track.phrases[:3]
    ranked = sorted(early, key=lambda p: -_ENTRY.get(p.role, 0.5))[:limit]
    if track.phrases[0] not in ranked:
        ranked.append(track.phrases[0])
    return ranked


def best_blend(a_track: Track, b_track: Track,
               weights: dict[str, float] | None = None) -> Blend | None:
    """The strongest transition from one specific record into another."""
    best = None
    for a in exit_phrases(a_track):
        for b in entry_phrases(b_track):
            cand = score_blend(a_track, a, b_track, b, weights)
            if cand and (best is None or cand.score > best.score):
                best = cand
    return best


def recommend(a_track: Track, crate: dict[int, Track], limit: int = 10,
              weights: dict[str, float] | None = None,
              exclude: set[int] | None = None) -> list[Blend]:
    """Rank the crate by how well each record follows the one playing."""
    skip = {a_track.id} | (exclude or set())
    out = []
    for tid, b in crate.items():
        if tid in skip:
            continue
        blend = best_blend(a_track, b, weights)
        if blend is not None:
            out.append(blend)
    out.sort(key=lambda x: -x.score)
    return out[:limit]

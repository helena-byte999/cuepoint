"""Listening to the room, so Cuepoint knows what is playing without being told.

Capture is deliberately vendor-neutral. Cuepoint never talks to Serato or
rekordbox; it takes an audio stream and identifies it against the library, so
the same code works for every DJ application, and for a pair of CDJs with no
computer involved at all.

Two ways in, both through ffmpeg, which the project already depends on:

    microphone      works immediately, no setup. A booth is loud and chroma is
                    level-independent, so this is far more robust than it
                    sounds -- it is how phone music recognition works.
    virtual device  BlackHole, Loopback or an audio interface loopback gives
                    the master output digitally, which is cleaner and survives
                    a noisy room or people talking over the record.

Once a track is identified the playhead is carried forward on a local clock
and only re-confirmed every few seconds. Correlating continuously would be
wasteful, and worse, it would make the position jitter; a clock is smooth and
the periodic re-match is what keeps it honest.

Position is not settled by one observation. Loop-based music repeats exactly,
so the first match usually offers several equally good positions (see
`fingerprint`). The listener keeps all of them, advances each on the clock,
and scores each against what actually arrives next. Hypotheses that predict
the wrong thing fall behind and are dropped; the survivor is the playhead.

How well that works, measured rather than hoped
-----------------------------------------------
Identity is reliable and fast: the right record is named within one window,
and a change of record is noticed within a single confirmation (~3s).

Absolute position is not always recoverable, and this is a property of the
music rather than a bug to be fixed later. On material with long exact
repeats the hypotheses can sit within 0.002 of each other for a minute at a
time, because the audio genuinely is the same; the listener then tracks
*stably* but can sit one loop early -- around 14 seconds on the track this
was measured against. Relative motion stays correct: the playhead advances in
lockstep with the record.

So `settled` is the flag to trust before treating the playhead as exact.
Recommendations do not depend on it -- `mix.recommend` works from the record's
identity -- but "bars remaining until the mix point" does, and callers should
say so rather than quietly implying a precision that is not there.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import audio, fingerprint

WINDOW_S = 8.0        # audio the matcher sees. Long enough to be distinctive,
                      # short enough that +/-8% pitch does not smear it.
RECHECK_S = 3.0       # how often to re-confirm the playhead
LOST_AFTER_S = 20.0   # give up on a track after this long with no confirmation


SEPARATION = 0.02     # score lead at which one hypothesis is declared the winner
DECAY = 0.7           # how much of its past agreement a hypothesis keeps
MAX_HYPOTHESES = 4
MISSES_TO_RELOCK = 2  # consecutive failed confirmations before searching again
PRUNE_GAP = 0.03      # drop a hypothesis once it trails the leader by this


@dataclass
class Hypothesis:
    """One candidate playhead, competing with the others to explain the audio.

    `score` is an exponential moving average of agreement, deliberately kept
    on the same 0..1 scale as a single observation so it can be compared
    against the match threshold directly. An accumulator that grew without
    bound would look confident purely because it had been running a while.
    """
    anchor_ms: int
    anchor_at: float
    score: float = 0.0
    last: float = 0.0

    def predict(self, rate: float, now: float | None = None) -> int:
        now = time.time() if now is None else now
        return int(self.anchor_ms + (now - self.anchor_at) * 1000.0 * rate)


@dataclass
class State:
    """What Cuepoint currently believes is playing."""
    track_id: int | None = None
    confidence: float = 0.0
    rate: float = 1.0
    settled: bool = False          # has one hypothesis won yet?
    last_seen: float = 0.0
    listening: bool = False
    error: str = ""
    hypotheses: list = field(default_factory=list)

    @property
    def playhead_ms(self) -> int:
        """Position now: the leading hypothesis, carried forward on the clock."""
        if self.track_id is None or not self.hypotheses:
            return 0
        return max(0, self.hypotheses[0].predict(self.rate))

    @property
    def stale(self) -> bool:
        return time.time() - self.last_seen > LOST_AFTER_S


def devices() -> list[tuple[int, str]]:
    """Audio inputs ffmpeg can open, as (index, name)."""
    proc = subprocess.run(
        [audio.FFMPEG, "-hide_banner", "-f", "avfoundation",
         "-list_devices", "true", "-i", ""],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = proc.stdout.decode("utf-8", "replace")
    out, seen_audio = [], False
    for line in text.splitlines():
        if "AVFoundation audio devices" in line:
            seen_audio = True
            continue
        if not seen_audio:
            continue
        m = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if m:
            out.append((int(m.group(1)), m.group(2).strip()))
    return out


def pick_device(prefer_loopback: bool = True) -> int:
    """Default input: a virtual/loopback device if one exists, else the mic.

    A loopback device carries the master output digitally and is always the
    better signal, so it wins automatically when the user has installed one.
    """
    found = devices()
    if not found:
        raise RuntimeError("ffmpeg reports no audio input devices")
    if prefer_loopback:
        for idx, name in found:
            if re.search(r"blackhole|loopback|soundflower|virtual|aggregate",
                         name, re.I):
                return idx
    return found[0][0]


class Listener:
    """Streams audio in a thread and keeps `state` current."""

    def __init__(self, index: fingerprint.Index, device: int | None = None,
                 window_s: float = WINDOW_S, recheck_s: float = RECHECK_S):
        self.index = index
        self.device = device
        self.window_s = window_s
        self.recheck_s = recheck_s
        self.state = State()
        self._buf = np.zeros(int(window_s * audio.SAMPLE_RATE), dtype=np.float32)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._misses = 0

    # -- capture ----------------------------------------------------------
    def _open(self) -> subprocess.Popen:
        dev = self.device if self.device is not None else pick_device()
        self.device = dev
        cmd = [audio.FFMPEG, "-v", "quiet", "-nostdin",
               "-f", "avfoundation", "-i", f":{dev}",
               "-ac", "1", "-ar", str(audio.SAMPLE_RATE),
               "-f", "f32le", "-"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)

    def _pump(self) -> None:
        """Read the stream into a ring buffer; match on a slower cadence."""
        try:
            self._proc = self._open()
        except Exception as exc:                     # noqa: BLE001
            self.state.error = f"cannot open audio input: {exc}"
            return

        chunk_frames = int(0.25 * audio.SAMPLE_RATE)
        nbytes = chunk_frames * 4
        self.state.listening = True
        last_check = 0.0

        while not self._stop.is_set():
            raw = self._proc.stdout.read(nbytes)
            if not raw:
                break
            block = np.frombuffer(raw, dtype=np.float32)
            with self._lock:
                n = len(block)
                if n >= len(self._buf):
                    self._buf[:] = block[-len(self._buf):]
                else:
                    self._buf[:-n] = self._buf[n:]
                    self._buf[-n:] = block

            now = time.time()
            if now - last_check >= self.recheck_s:
                last_check = now
                self._identify()

        self.state.listening = False

    # -- identification ---------------------------------------------------
    def _chroma_now(self) -> np.ndarray | None:
        with self._lock:
            window = self._buf.copy()
        peak = float(np.abs(window).max())
        if peak < 1e-4:            # dead room: don't burn a correlation on it
            return None
        return fingerprint.chroma_series(window / peak)

    def _relock(self, query: np.ndarray) -> None:
        """No idea what is playing: search the whole library."""
        cands = self.index.candidates(query, top_k=MAX_HYPOTHESES)
        if not cands or not cands[0].solid:
            if self.state.stale:
                self.state.track_id = None
                self.state.hypotheses = []
                self.state.confidence = 0.0
            return
        now = time.time()
        s = self.state
        s.track_id = cands[0].track_id
        s.rate = cands[0].rate
        s.confidence = cands[0].confidence
        s.settled = len(cands) == 1
        s.last_seen = now
        s.error = ""
        s.hypotheses = [Hypothesis(c.offset_ms, now, c.confidence, c.confidence)
                        for c in cands]
        self._misses = 0

    def _confirm(self, query: np.ndarray) -> None:
        """We think we know the record: check each candidate playhead."""
        s = self.state
        now = time.time()
        alive = []
        for h in s.hypotheses:
            m = self.index.match_local(query, s.track_id, h.predict(s.rate, now),
                                       radius_ms=3000, rate=s.rate)
            if m is None:
                continue
            h.last = m.confidence
            h.score = DECAY * h.score + (1.0 - DECAY) * m.confidence
            h.anchor_ms, h.anchor_at = m.offset_ms, now
            alive.append(h)

        # Judge "is this still the same record?" on what we just heard, not on
        # the average. A long-running average stays high for a while after the
        # DJ has already moved on, and the whole point is to notice quickly.
        if not alive or max(h.last for h in alive) < fingerprint.MIN_CONFIDENCE:
            self._misses += 1
            if self._misses >= MISSES_TO_RELOCK:
                self._relock(query)
            return
        self._misses = 0

        # Hysteresis. While a record is looping, every hypothesis explains the
        # audio equally well and their scores sit within a thousandth of each
        # other; re-sorting each time would make the reported playhead hop
        # between identical repeats. The incumbent therefore keeps the lead
        # until a rival genuinely beats it.
        leader = s.hypotheses[0] if s.hypotheses else alive[0]
        if leader not in alive:
            leader = max(alive, key=lambda h: h.score)
        best = max(alive, key=lambda h: h.score)
        if best is not leader and best.score - leader.score > SEPARATION:
            leader = best

        # Once the arrangement moves on, the repeats stop explaining it and
        # those hypotheses fall away on their own.
        alive = [h for h in alive if h.score >= leader.score - PRUNE_GAP]
        alive.sort(key=lambda h: (h is not leader, -h.score))

        s.hypotheses = alive[:MAX_HYPOTHESES]
        s.settled = len(s.hypotheses) == 1
        s.confidence = leader.score
        s.last_seen = now

    def _identify(self) -> None:
        query = self._chroma_now()
        if query is None:
            return
        if self.state.track_id is None or not self.state.hypotheses:
            self._relock(query)
        else:
            self._confirm(query)

    # -- lifecycle --------------------------------------------------------
    def start(self) -> "Listener":
        # Resolved here rather than in the capture thread: callers print the
        # device straight after start(), and racing that read reports "None"
        # for a listener that is in fact about to open the right input.
        if self.device is None:
            self.device = pick_device()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._proc:
            self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=2)

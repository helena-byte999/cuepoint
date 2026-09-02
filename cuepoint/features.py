"""Per-phrase acoustic features.

For every phrase rekordbox identified, we compute the things that decide
whether two records will sit on top of each other:

    groove_low   kick/bass rhythm over one bar, 24 slots
    groove_full  whole-mix rhythm over one bar, 24 slots
    chroma       12 pitch classes, for harmonic distance
    kick_hz      fundamental of the kick drum
    energy       loudness, in dB
    vocal        how much sustained pitched content sits in the vocal band

Everything is bar-synchronous, using rekordbox's own beat grid, so patterns
from tracks at different tempos are directly comparable.

Note on `vocal`: this is a heuristic (harmonic/percussive separation plus a
band-limited energy ratio), not a trained vocal detector. It reliably
separates "instrumental" from "someone is singing", which is what the scorer
needs; it is not a transcription and it will not catch a heavily processed
vocal chop. Real transcription would slot in here.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from . import audio

# Two resolutions: fine hop for rhythm (we need to place onsets inside a bar),
# long window for pitch (we need to resolve semitones in the low mids).
RHYTHM_NFFT, RHYTHM_HOP = 1024, 256      # 11.6 ms steps
PITCH_NFFT, PITCH_HOP = 4096, 1024       # 5.4 Hz bins

SLOTS = 24        # subdivisions of a bar; divisible by 3 and 4, so it can
                  # represent both triplet and sixteenth-note grooves

LOW_BAND = (20.0, 160.0)      # kick and sub
VOCAL_BAND = (200.0, 3500.0)  # where a human voice lives
CHROMA_BAND = (130.0, 2093.0)  # C3..C7; below this, FFT bins are too coarse

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler profiles, used to name the key for display only. The
# scorer uses the raw chroma vector, which carries far more information than
# a single label.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Camelot wheel: (pitch class, is_minor) -> code
_CAMELOT_MAJOR = {0: "8B", 1: "3B", 2: "10B", 3: "5B", 4: "12B", 5: "7B",
                  6: "2B", 7: "9B", 8: "4B", 9: "11B", 10: "6B", 11: "1B"}
_CAMELOT_MINOR = {0: "5A", 1: "12A", 2: "7A", 3: "2A", 4: "9A", 5: "4A",
                  6: "11A", 7: "6A", 8: "1A", 9: "8A", 10: "3A", 11: "10A"}


@dataclass
class PhraseFeature:
    idx: int
    groove_low: np.ndarray
    groove_full: np.ndarray
    chroma: np.ndarray
    kick_hz: float
    energy_db: float
    vocal: float


def _chroma_matrix(freqs: np.ndarray) -> np.ndarray:
    """Map FFT bins onto 12 pitch classes, restricted to a usable range."""
    lo, hi = CHROMA_BAND
    usable = (freqs >= lo) & (freqs <= hi)
    midi = np.zeros_like(freqs)
    midi[usable] = 69 + 12 * np.log2(freqs[usable] / 440.0)
    pc = np.round(midi).astype(int) % 12

    m = np.zeros((12, len(freqs)), dtype=np.float32)
    m[pc[usable], np.where(usable)[0]] = 1.0
    return m


def estimate_key(chroma: np.ndarray) -> tuple[str, str]:
    """Return (key name, Camelot code) from a 12-vector."""
    if chroma.sum() <= 0:
        return "?", "?"
    c = chroma / chroma.sum()
    best, best_score, best_minor = 0, -np.inf, False
    for shift in range(12):
        rolled = np.roll(c, -shift)
        for profile, minor in ((_MAJOR, False), (_MINOR, True)):
            score = float(np.corrcoef(rolled, profile)[0, 1])
            if score > best_score:
                best, best_score, best_minor = shift, score, minor
    name = PITCH_CLASSES[best] + ("m" if best_minor else "")
    code = (_CAMELOT_MINOR if best_minor else _CAMELOT_MAJOR)[best]
    return name, code


def _bar_groove(env: np.ndarray, times: np.ndarray,
                bar_edges: np.ndarray) -> np.ndarray:
    """Average one bar's worth of onset activity across many bars.

    Each bar is divided into SLOTS equal parts regardless of tempo, so the
    resulting vector is a tempo-independent description of the groove.
    """
    acc = np.zeros(SLOTS, dtype=np.float64)
    used = 0
    for start, end in zip(bar_edges[:-1], bar_edges[1:]):
        if end <= start:
            continue
        edges = start + (end - start) * np.arange(SLOTS + 1) / SLOTS
        lo = np.searchsorted(times, edges[:-1], "left")
        hi = np.searchsorted(times, edges[1:], "left")
        if np.any(hi <= lo):
            continue
        # Max, not mean: an onset is an event, and averaging over a slot that
        # is mostly silence would wash it out.
        acc += [env[a:b].max() for a, b in zip(lo, hi)]
        used += 1
    if used == 0:
        return acc.astype(np.float32)
    acc /= used
    peak = acc.max()
    return (acc / peak).astype(np.float32) if peak > 0 else acc.astype(np.float32)


def analyse(path: str, beat_times_ms: np.ndarray, downbeat_offset: int,
            phrases: list[tuple[int, int, int]]) -> list[PhraseFeature]:
    """Compute features for each phrase of one track.

    `phrases` is a list of (idx, start_beat, end_beat), 1-based beat indices
    into the grid, matching what `library` stored.
    """
    y = audio.decode(path)
    if y.size < audio.SAMPLE_RATE:
        raise audio.DecodeError(f"{Path(path).name}: too short to analyse")

    beat_s = beat_times_ms / 1000.0

    # --- rhythm resolution -------------------------------------------------
    spec_r = audio.spectrogram(y, RHYTHM_NFFT, RHYTHM_HOP)
    freqs_r = audio.fft_freqs(RHYTHM_NFFT)
    times_r = audio.frame_times(spec_r.shape[1], RHYTHM_HOP, RHYTHM_NFFT)
    env_low = audio.onset_envelope(spec_r, freqs_r, LOW_BAND)
    env_full = audio.onset_envelope(spec_r, freqs_r)

    # Align the decoded audio to rekordbox's grid before measuring any
    # rhythm, or every groove template comes out rotated.
    times_r = times_r + audio.grid_offset(env_low, times_r, beat_s)

    # --- pitch resolution --------------------------------------------------
    spec_p = audio.spectrogram(y, PITCH_NFFT, PITCH_HOP)
    freqs_p = audio.fft_freqs(PITCH_NFFT)
    times_p = audio.frame_times(spec_p.shape[1], PITCH_HOP, PITCH_NFFT)
    chroma_map = _chroma_matrix(freqs_p)
    chroma_frames = chroma_map @ spec_p            # (12, frames)

    # --- harmonic / percussive split, for the vocal proxy ------------------
    # A sustained note is stable over time and narrow in frequency; a drum hit
    # is the opposite. Median-filtering each way separates them cheaply.
    harm = median_filter(spec_p, size=(1, 17), mode="nearest")
    perc = median_filter(spec_p, size=(17, 1), mode="nearest")
    denom = harm**2 + perc**2 + 1e-9
    harm_mask = (harm**2) / denom
    vocal_bins = (freqs_p >= VOCAL_BAND[0]) & (freqs_p < VOCAL_BAND[1])
    harm_energy = (spec_p[vocal_bins] * harm_mask[vocal_bins]).sum(axis=0)
    total_energy = spec_p.sum(axis=0) + 1e-9

    out: list[PhraseFeature] = []
    for idx, start_beat, end_beat in phrases:
        s = float(beat_s[min(start_beat - 1, len(beat_s) - 1)])
        e = float(beat_s[min(end_beat - 1, len(beat_s) - 1)])
        if e <= s:
            continue

        # Bar edges inside this phrase, taken from the real grid.
        first = start_beat - 1 + ((downbeat_offset - (start_beat - 1)) % 4)
        bar_beats = np.arange(first, end_beat, 4)
        bar_edges = beat_s[bar_beats[bar_beats < len(beat_s)]]
        if len(bar_edges) < 2:
            bar_edges = np.array([s, e])

        g_low = _bar_groove(env_low, times_r, bar_edges)
        g_full = _bar_groove(env_full, times_r, bar_edges)

        pm = (times_p >= s) & (times_p < e)
        if not pm.any():
            pm = np.zeros_like(times_p, dtype=bool)
            pm[min(int(s / (PITCH_HOP / audio.SAMPLE_RATE)), len(pm) - 1)] = True

        chroma = chroma_frames[:, pm].mean(axis=1)
        norm = np.linalg.norm(chroma)
        chroma = chroma / norm if norm > 0 else chroma

        # Loudness of the raw signal over the phrase.
        i0, i1 = int(s * audio.SAMPLE_RATE), int(e * audio.SAMPLE_RATE)
        seg = y[i0:min(i1, len(y))]
        rms = float(np.sqrt(np.mean(seg**2))) if seg.size else 0.0
        energy_db = 20 * np.log10(rms + 1e-9)

        vocal = float(np.median(harm_energy[pm] / total_energy[pm]))
        kick_hz = _kick_fundamental(seg)

        out.append(PhraseFeature(idx=idx, groove_low=g_low, groove_full=g_full,
                                 chroma=chroma.astype(np.float32),
                                 kick_hz=kick_hz, energy_db=energy_db,
                                 vocal=vocal))
    return out


def _kick_fundamental(seg: np.ndarray) -> float:
    """Dominant frequency below 100 Hz -- effectively the kick's pitch."""
    if seg.size < 4096:
        return 0.0
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / audio.SAMPLE_RATE)
    band = (freqs >= 35) & (freqs <= 100)
    if not band.any():
        return 0.0
    return float(freqs[band][np.argmax(spec[band])])


def save(path: str | Path, feats: list[PhraseFeature]) -> None:
    """Cache one track's features so re-runs are instant."""
    if not feats:
        return
    np.savez_compressed(
        path,
        idx=np.array([f.idx for f in feats]),
        groove_low=np.stack([f.groove_low for f in feats]),
        groove_full=np.stack([f.groove_full for f in feats]),
        chroma=np.stack([f.chroma for f in feats]),
        kick_hz=np.array([f.kick_hz for f in feats]),
        energy_db=np.array([f.energy_db for f in feats]),
        vocal=np.array([f.vocal for f in feats]),
    )


def load(path: str | Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}

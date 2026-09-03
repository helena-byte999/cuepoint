"""Audio decoding and the small amount of DSP we need.

Deliberately built on numpy/scipy alone. The usual choice here is librosa,
which drags in numba and a newer Python than this machine has; everything we
actually use is a few dozen lines, so we own it instead.

Decoding goes through ffmpeg, which is already installed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050

def _find_ffmpeg() -> str:
    """Locate ffmpeg: the system one first, then the bundled fallback.

    A DJ should not have to install a binary by hand before the tool runs, so
    imageio-ffmpeg is a dependency and supplies a working build everywhere.
    A system ffmpeg still wins when it exists -- it is usually newer and
    carries wider codec coverage than any bundled build.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                        # noqa: BLE001
        return str(Path.home() / "bin/ffmpeg")


FFMPEG = _find_ffmpeg()


class DecodeError(RuntimeError):
    pass


def decode(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio file to mono float32 at `sr`.

    ffmpeg handles the format zoo (mp3/m4a/flac/wav/aiff) and the resampling,
    and streams raw floats to stdout so nothing hits disk.
    """
    cmd = [
        FFMPEG, "-v", "quiet", "-nostdin",
        "-i", str(path),
        "-ac", "1",            # mono
        "-ar", str(sr),
        "-f", "f32le",         # raw 32-bit float
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        msg = proc.stderr.decode("utf-8", "replace").strip()[:200]
        raise DecodeError(f"ffmpeg failed for {Path(path).name}: {msg}")
    y = np.frombuffer(proc.stdout, dtype=np.float32)
    peak = float(np.abs(y).max()) if y.size else 0.0
    # Normalise so loudness differences between rips don't leak into the
    # spectral features. Real loudness is measured separately, per phrase.
    return y / peak if peak > 0 else y


def frame(y: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Split a signal into overlapping frames without copying."""
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    n = 1 + (len(y) - n_fft) // hop
    stride = y.strides[0]
    return np.lib.stride_tricks.as_strided(
        y, shape=(n, n_fft), strides=(hop * stride, stride), writeable=False
    )


_WINDOWS: dict[int, np.ndarray] = {}


def _hann(n: int) -> np.ndarray:
    if n not in _WINDOWS:
        _WINDOWS[n] = np.hanning(n).astype(np.float32)
    return _WINDOWS[n]


def spectrogram(y: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Magnitude spectrogram, shape (1 + n_fft/2, frames)."""
    frames = frame(y, n_fft, hop) * _hann(n_fft)
    return np.abs(np.fft.rfft(frames, axis=1)).T.astype(np.float32)


def frame_times(n_frames: int, hop: int, n_fft: int,
                sr: int = SAMPLE_RATE) -> np.ndarray:
    """Centre time, in seconds, of each spectrogram frame.

    The centre, not the start: a frame describes the whole window, so timing
    it from its leading edge reports every event half a window early.
    """
    return (np.arange(n_frames) * hop + n_fft / 2.0) / sr


def grid_offset(env: np.ndarray, times: np.ndarray,
                beat_s: np.ndarray, max_frac: float = 0.25) -> float:
    """Seconds to add to `times` to line onsets up with the beat grid.

    Even after centring the analysis windows, decoded audio sits a little
    behind rekordbox's grid -- MP3 encoder delay is the usual culprit, and it
    varies by encoder, so it can't be a constant. We measure it instead: take
    the strong onsets, look at how far each sits from its nearest grid beat,
    and use the median. The median ignores syncopated hits that genuinely fall
    off the grid, which matters for broken-kick material.
    """
    if len(beat_s) < 8 or env.size == 0:
        return 0.0
    beat_dur = float(np.median(np.diff(beat_s)))
    if beat_dur <= 0:
        return 0.0

    strong = times[env >= np.percentile(env, 97)]
    strong = strong[(strong > beat_s[0]) & (strong < beat_s[-1])]
    if strong.size < 8:
        return 0.0

    i = np.clip(np.searchsorted(beat_s, strong), 1, len(beat_s) - 1)
    prev, nxt = beat_s[i - 1], beat_s[i]
    delta = np.where(strong - prev < nxt - strong, strong - prev, strong - nxt)

    shift = -float(np.median(delta))
    return float(np.clip(shift, -max_frac * beat_dur, max_frac * beat_dur))


def fft_freqs(n_fft: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.fft.rfftfreq(n_fft, 1.0 / sr)


def onset_envelope(spec: np.ndarray, freqs: np.ndarray,
                   band: tuple[float, float] | None = None) -> np.ndarray:
    """Spectral flux: how much energy *appeared* since the previous frame.

    Only increases count -- energy dying away is not an onset. Magnitudes are
    log-compressed first so a quiet hi-hat still registers against a loud kick,
    which matters because we care about the pattern, not the levels.
    """
    s = spec
    if band is not None:
        lo, hi = band
        s = s[(freqs >= lo) & (freqs < hi)]
    s = np.log1p(1000.0 * s)
    flux = np.diff(s, axis=1, prepend=s[:, :1])
    env = np.maximum(flux, 0).sum(axis=0)
    peak = env.max()
    return (env / peak) if peak > 0 else env

"""Parser for rekordbox ANLZ analysis files (.DAT / .EXT / .2EX).

These are the files rekordbox writes when it analyses a track. They are the
same format burned onto a USB stick for CDJs, and they already contain the
beat grid, cue points and phrase structure -- so we never need to recompute
any of it.

Layout: a `PMAI` container header, then a flat sequence of tags. Every tag
starts with a 4-byte magic, a u32 header length and a u32 total length
(header included), so an unknown tag can always be skipped. All values are
big-endian.

Tags we read:
    PPTH  original file path (UTF-16BE)
    PQTZ  beat grid: beat-in-bar, tempo, time
    PCOB  cue points (.DAT = memory cues, .EXT = hot cues)
    PCO2  cue points, extended form with names and colours
    PSSI  phrase structure (intro / build / chorus / outro)

Everything here is read-only; we never write into rekordbox's own files.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

MAGIC = b"PMAI"


# --------------------------------------------------------------------------
# phrase kinds
# --------------------------------------------------------------------------
# rekordbox picks one of three "moods" per track, and the phrase kind numbers
# mean different things in each. The mapping below was confirmed against all
# 431 analysed tracks in this collection: in mood 1 every track starts with
# kind 1 and ends with kind 6, and kind 4 never appears at all -- which is the
# signature of the high-mood layout.

MOOD_HIGH, MOOD_MID, MOOD_LOW = 1, 2, 3

# Normalised roles, so the scorer doesn't care which mood a track used.
INTRO, BUILD, PEAK, BREAKDOWN, VERSE, BRIDGE, OUTRO = (
    "intro", "build", "peak", "breakdown", "verse", "bridge", "outro",
)

_HIGH_KINDS = {1: INTRO, 2: BUILD, 3: BREAKDOWN, 5: PEAK, 6: OUTRO}
_MID_KINDS = {
    1: INTRO, 2: VERSE, 3: VERSE, 4: VERSE, 5: VERSE, 6: VERSE, 7: VERSE,
    8: BRIDGE, 9: PEAK, 10: OUTRO,
}


def phrase_role(mood: int, kind: int) -> str:
    """Map a (mood, kind) pair to a mood-independent structural role."""
    table = _HIGH_KINDS if mood == MOOD_HIGH else _MID_KINDS
    return table.get(kind, VERSE)


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------

@dataclass
class Beat:
    number: int      # position in the bar, 1-4
    tempo: float     # BPM at this beat (rekordbox stores BPM*100)
    time_ms: int


@dataclass
class Cue:
    kind: str        # "memory" | "hot"
    time_ms: int
    hot_number: int  # 0 for memory cues, else 1=A, 2=B, ...
    comment: str = ""
    color: tuple[int, int, int] | None = None


@dataclass
class Phrase:
    index: int
    beat: int        # 1-based beat number where the phrase starts
    kind: int        # raw rekordbox kind
    role: str        # normalised role (intro/build/peak/...)


@dataclass
class Analysis:
    path: str = ""
    beats: list[Beat] = field(default_factory=list)
    cues: list[Cue] = field(default_factory=list)
    phrases: list[Phrase] = field(default_factory=list)
    mood: int = 0
    end_beat: int = 0

    @property
    def tempo(self) -> float:
        """Median beat tempo. Median, not mean, so a couple of bad grid
        markers at the very start or end can't drag the value."""
        if not self.beats:
            return 0.0
        t = sorted(b.tempo for b in self.beats)
        return t[len(t) // 2]

    @property
    def downbeat_offset(self) -> int:
        """Index of the first true downbeat in the grid.

        rekordbox does not always start its grid on beat 1 of a bar -- 91 of
        the 429 analysed tracks here start on beat 2, 3 or 4. Bar numbers must
        be counted from the first beat whose number is 1, or every bar we
        report is off by up to three beats.
        """
        for i, b in enumerate(self.beats):
            if b.number == 1:
                return i
        return 0

    def beat_to_ms(self, beat: int) -> int:
        """1-based index into the beat grid -> milliseconds."""
        if not self.beats:
            return 0
        return self.beats[max(0, min(beat - 1, len(self.beats) - 1))].time_ms

    def beat_to_bar(self, beat: int) -> int:
        """1-based beat index -> 1-based musical bar number."""
        return (beat - 1 - self.downbeat_offset) // 4 + 1

    def bar_to_ms(self, bar: int) -> int:
        return self.beat_to_ms((bar - 1) * 4 + 1 + self.downbeat_offset)


# --------------------------------------------------------------------------
# container walking
# --------------------------------------------------------------------------

def _tags(data: bytes) -> Iterator[tuple[str, bytes]]:
    """Yield (magic, body) for each tag, skipping anything unrecognised."""
    if len(data) < 12 or data[:4] != MAGIC:
        return
    offset = struct.unpack_from(">I", data, 4)[0]
    while offset + 12 <= len(data):
        magic = data[offset:offset + 4]
        if not magic.isalnum():
            break
        total = struct.unpack_from(">I", data, offset + 8)[0]
        # A zero or negative length would loop forever; a length past the end
        # means the file is truncated. Either way, stop.
        if total < 12 or offset + total > len(data):
            break
        yield magic.decode("ascii"), data[offset:offset + total]
        offset += total


def _utf16_string(body: bytes, offset: int, length: int) -> str:
    raw = body[offset:offset + length]
    return raw.decode("utf-16-be", errors="replace").rstrip("\x00")


# --------------------------------------------------------------------------
# individual tags
# --------------------------------------------------------------------------

def _parse_ppth(body: bytes) -> str:
    length = struct.unpack_from(">I", body, 12)[0]
    return _utf16_string(body, 16, length)


def _parse_pqtz(body: bytes) -> list[Beat]:
    count = struct.unpack_from(">I", body, 20)[0]
    beats = []
    for i in range(count):
        off = 24 + i * 8
        if off + 8 > len(body):
            break
        number, tempo, time_ms = struct.unpack_from(">HHI", body, off)
        beats.append(Beat(number=number, tempo=tempo / 100.0, time_ms=time_ms))
    return beats


def _subtags(body: bytes, magic: bytes) -> Iterator[bytes]:
    """Walk the child records inside a cue list.

    We scan for the child magic rather than trusting the parent's count
    field, whose exact offset differs between format revisions. Walking the
    children is unambiguous and costs nothing.
    """
    off = struct.unpack_from(">I", body, 4)[0]
    while off + 12 <= len(body):
        if body[off:off + 4] != magic:
            break
        total = struct.unpack_from(">I", body, off + 8)[0]
        if total < 12 or off + total > len(body):
            break
        yield body[off:off + total]
        off += total


# NOTE: the two cue parsers below are written to the documented layout but
# are UNVERIFIED against real data -- every one of the 2810 cue tags in this
# collection is empty, because no cues have ever been saved. Re-check the
# offsets against a real file before relying on cue output.

def _parse_pcob(body: bytes, kind: str) -> list[Cue]:
    """Plain cue list, holding PCPT records."""
    cues = []
    for rec in _subtags(body, b"PCPT"):
        if len(rec) < 36:
            continue
        hot = struct.unpack_from(">I", rec, 12)[0]
        time_ms = struct.unpack_from(">I", rec, 32)[0]
        cues.append(Cue(kind=kind, time_ms=time_ms, hot_number=hot))
    return cues


def _parse_pco2(body: bytes, kind: str) -> list[Cue]:
    """Extended cue list: adds a name and an explicit RGB colour."""
    cues = []
    for rec in _subtags(body, b"PCP2"):
        if len(rec) < 24:
            continue
        hot = struct.unpack_from(">I", rec, 12)[0]
        time_ms = struct.unpack_from(">I", rec, 20)[0]
        color = None
        comment = ""
        if len(rec) >= 48:
            clen = struct.unpack_from(">I", rec, 44)[0]
            if 0 < clen <= len(rec) - 48:
                comment = _utf16_string(rec, 48, clen)
            tail = 48 + clen
            if tail + 7 <= len(rec):
                color = (rec[tail + 4], rec[tail + 5], rec[tail + 6])
        cues.append(Cue(kind=kind, time_ms=time_ms, hot_number=hot,
                        comment=comment, color=color))
    return cues


def _parse_pssi(body: bytes) -> tuple[list[Phrase], int, int]:
    """Phrase structure. Returns (phrases, mood, end_beat).

    Header is 32 bytes: u32 entry size at 12, u16 entry count at 16, u16 mood
    at 18, u16 end beat at 26. Verified: (tag_len - header_len) / entry_size
    equals the entry count for every analysed track in this collection.
    """
    header, total = struct.unpack_from(">II", body, 4)
    entry_len = struct.unpack_from(">I", body, 12)[0]
    count, mood = struct.unpack_from(">HH", body, 16)
    end_beat = struct.unpack_from(">H", body, 26)[0]
    if entry_len == 0:
        return [], mood, end_beat

    phrases = []
    for i in range(count):
        off = header + i * entry_len
        if off + 6 > len(body):
            break
        index, beat, kind = struct.unpack_from(">HHH", body, off)
        phrases.append(Phrase(index=index, beat=beat, kind=kind,
                              role=phrase_role(mood, kind)))
    return phrases, mood, end_beat


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def load(dat_path: str | Path) -> Analysis:
    """Read a track's analysis, merging the .DAT and .EXT siblings.

    The .DAT holds the beat grid and memory cues; the .EXT adds hot cues and
    the phrase structure. Either may be missing.
    """
    result = Analysis()
    if not str(dat_path).strip():
        # rekordbox leaves this blank for tracks it never analysed.
        return result
    dat = Path(dat_path)

    if dat.is_file():
        data = dat.read_bytes()
        for magic, body in _tags(data):
            if magic == "PPTH":
                result.path = _parse_ppth(body)
            elif magic == "PQTZ":
                result.beats = _parse_pqtz(body)
            elif magic == "PCOB":
                result.cues += _parse_pcob(body, "memory")

    ext = dat.with_suffix(".EXT")
    if ext.is_file():
        data = ext.read_bytes()
        seen_pco2 = False
        for magic, body in _tags(data):
            if magic == "PPTH" and not result.path:
                result.path = _parse_ppth(body)
            elif magic == "PQT2" and not result.beats:
                result.beats = _parse_pqtz(body)
            elif magic == "PCO2":
                # PCO2 supersedes PCOB where both exist -- it carries names
                # and colours. Replace rather than append so cues aren't
                # counted twice.
                if not seen_pco2:
                    result.cues = [c for c in result.cues if c.kind != "hot"]
                    seen_pco2 = True
                result.cues += _parse_pco2(body, "hot")
            elif magic == "PSSI":
                result.phrases, result.mood, result.end_beat = _parse_pssi(body)

    result.cues.sort(key=lambda c: c.time_ms)
    return result

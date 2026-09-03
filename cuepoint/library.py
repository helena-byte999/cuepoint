"""Build Cuepoint's own database from the rekordbox collection.

rekordbox keeps its real library in `master.db`, which is SQLCipher-encrypted
and which we never touch. Everything we need is available in the clear:

    networkAnalyze6.db   plain SQLite, maps track -> file path -> ANLZ path
    share/PIONEER/USBANLZ/**   the analysis files themselves

Both are opened strictly read-only. Cuepoint writes only to its own
`cuepoint.db`, so nothing here can corrupt a collection.
"""

from __future__ import annotations

import os
import sqlite3
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from . import anlz, paths

ANALYZE_NAME = "networkAnalyze6.db"


def rekordbox_dirs() -> list[Path]:
    """Everywhere rekordbox is known to keep its analysis data.

    macOS and Windows put it in different places, and a user can move it, so
    CUEPOINT_REKORDBOX overrides the search entirely.
    """
    env = os.environ.get("CUEPOINT_REKORDBOX")
    if env:
        return [Path(env).expanduser()]
    home = Path.home()
    cand = [
        home / "Library/Pioneer/rekordbox",                   # macOS
        Path(os.environ.get("APPDATA", home / "AppData/Roaming")) / "Pioneer/rekordbox",
        home / "AppData/Roaming/Pioneer/rekordbox",           # Windows
    ]
    # APPDATA usually resolves to the third entry, and listing the same folder
    # twice in a "looked in" error just makes it look broken.
    seen, out = set(), []
    for d in cand:
        if str(d) not in seen:
            seen.add(str(d)); out.append(d)
    return out


def find_rekordbox() -> Path | None:
    """The first rekordbox directory that actually holds an analysis db."""
    for d in rekordbox_dirs():
        try:
            if (d / ANALYZE_NAME).exists():
                return d
        except OSError:
            continue
    return None


def analyze_db() -> Path:
    d = find_rekordbox()
    if d is None:
        searched = "\n  ".join(str(x) for x in rekordbox_dirs())
        raise FileNotFoundError(
            "Could not find a rekordbox collection. Looked in:\n  "
            + searched
            + "\n\nInstall rekordbox and analyse your tracks, or point Cuepoint "
              "at the right folder with CUEPOINT_REKORDBOX=/path/to/rekordbox"
        )
    return d / ANALYZE_NAME

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL,
    filename    TEXT NOT NULL,
    title       TEXT NOT NULL,
    folder      TEXT,
    present     INTEGER NOT NULL,   -- audio file still on disk?
    duration_ms INTEGER,
    tempo       REAL,
    beat_count  INTEGER,
    mood        INTEGER,
    analyzed    INTEGER NOT NULL    -- has usable beat grid + phrases?
);
CREATE TABLE IF NOT EXISTS beats (
    track_id INTEGER NOT NULL,
    idx      INTEGER NOT NULL,
    number   INTEGER NOT NULL,
    tempo    REAL    NOT NULL,
    time_ms  INTEGER NOT NULL,
    PRIMARY KEY (track_id, idx)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS phrases (
    track_id   INTEGER NOT NULL,
    idx        INTEGER NOT NULL,
    role       TEXT    NOT NULL,
    kind       INTEGER NOT NULL,
    start_beat INTEGER NOT NULL,
    end_beat   INTEGER NOT NULL,
    start_bar  INTEGER NOT NULL,
    bars       INTEGER NOT NULL,
    start_ms   INTEGER NOT NULL,
    end_ms     INTEGER NOT NULL,
    PRIMARY KEY (track_id, idx)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS cues (
    track_id INTEGER NOT NULL,
    kind     TEXT NOT NULL,
    hot      INTEGER NOT NULL,
    time_ms  INTEGER NOT NULL,
    comment  TEXT
);
CREATE INDEX IF NOT EXISTS phrases_role ON phrases(role);
CREATE INDEX IF NOT EXISTS tracks_present ON tracks(present, analyzed);
"""

@dataclass
class Source:
    track_id: int
    audio_path: str
    anlz_path: str
    duration_ms: int


def _readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _clean_title(filename: str) -> str:
    """Strip the junk that download sites bolt onto filenames.

    These names are genuinely messy in the wild, e.g.
    "[SPOTDOWNLOADER.COM] Return Of The Mack - Remix.mp3" or
    "... - SoundLoadMate.com.mp3". A tidy title makes the UI readable; it is
    never used for matching, so an imperfect result is harmless.
    """
    name = Path(filename).stem
    for junk in ("[SPOTDOWNLOADER.COM]", "SPOTDOWNLOADER.COM",
                 "- SoundLoadMate.com", "SoundLoadMate.com",
                 "(320 KBps)", "(Official Audio)", "[Official Audio]"):
        name = name.replace(junk, "")
    # Trailing YouTube-style id, e.g. "[0GdQFoCKf7U]"
    if name.rstrip().endswith("]") and "[" in name:
        head, _, tail = name.rstrip()[:-1].rpartition("[")
        if len(tail) == 11 and " " not in tail:
            name = head
    return " ".join(name.split()).strip(" -_")


def sources() -> list[Source]:
    """Every analysed track rekordbox knows about, with decoded paths."""
    con = _readonly(analyze_db())
    try:
        rows = con.execute(
            "SELECT SongID, SongFilePath, AnalyzeFilePath, Duration "
            "FROM manage_tbl"
        ).fetchall()
    finally:
        con.close()

    out = []
    for track_id, song_path, anlz_path, duration in rows:
        # rekordbox percent-encodes these, including non-ASCII characters.
        audio = urllib.parse.unquote(song_path or "")
        if not audio:
            continue
        # Duration is already milliseconds, despite the bare name -- checked
        # against the beat grids, where the last beat of a track lands within
        # a few ms of this value.
        out.append(Source(track_id=track_id, audio_path=audio,
                          anlz_path=anlz_path or "",
                          duration_ms=duration or 0))
    return out


def build(db_path: str | Path, verbose: bool = True) -> dict:
    """Parse every analysis file and write Cuepoint's database.

    Returns a summary dict; safe to re-run, it rebuilds from scratch.
    """
    db_path = Path(db_path)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    for table in ("tracks", "beats", "phrases", "cues"):
        con.execute(f"DELETE FROM {table}")

    stats = {"total": 0, "present": 0, "with_grid": 0,
             "with_phrases": 0, "usable": 0, "failed": 0}

    for src in sources():
        stats["total"] += 1
        try:
            a = anlz.load(src.anlz_path)
        except Exception:
            stats["failed"] += 1
            continue

        present = os.path.exists(src.audio_path)
        has_grid = len(a.beats) > 0
        has_phrases = len(a.phrases) > 0
        # "usable" means we can actually reason about it: we need the grid to
        # convert bars to time, and the phrases to pick mix points.
        usable = present and has_grid and has_phrases

        stats["present"] += present
        stats["with_grid"] += has_grid
        stats["with_phrases"] += has_phrases
        stats["usable"] += usable

        filename = os.path.basename(src.audio_path)
        folder = os.path.basename(os.path.dirname(src.audio_path))
        con.execute(
            "INSERT OR REPLACE INTO tracks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (src.track_id, src.audio_path, filename, _clean_title(filename),
             folder, int(present), src.duration_ms, a.tempo, len(a.beats),
             a.mood, int(has_grid and has_phrases)),
        )

        con.executemany(
            "INSERT OR REPLACE INTO beats VALUES (?,?,?,?,?)",
            [(src.track_id, i, b.number, b.tempo, b.time_ms)
             for i, b in enumerate(a.beats)],
        )

        # Phrases store only a start; the end is the next phrase's start, and
        # the last one runs to end_beat. Precomputing both ends here keeps the
        # scorer free of off-by-one handling.
        n_beats = len(a.beats)
        for i, p in enumerate(a.phrases):
            nxt = a.phrases[i + 1].beat if i + 1 < len(a.phrases) else (
                a.end_beat or n_beats)
            end_beat = max(p.beat, nxt)
            con.execute(
                "INSERT OR REPLACE INTO phrases VALUES (?,?,?,?,?,?,?,?,?,?)",
                (src.track_id, i, p.role, p.kind, p.beat, end_beat,
                 a.beat_to_bar(p.beat),
                 max(1, (end_beat - p.beat) // 4),
                 a.beat_to_ms(p.beat), a.beat_to_ms(end_beat)),
            )

        con.executemany(
            "INSERT INTO cues VALUES (?,?,?,?,?)",
            [(src.track_id, c.kind, c.hot_number, c.time_ms, c.comment)
             for c in a.cues],
        )

    con.commit()
    con.close()

    if verbose:
        print(f"  tracks known to rekordbox : {stats['total']}")
        print(f"  audio file still on disk  : {stats['present']}")
        print(f"  with beat grid            : {stats['with_grid']}")
        print(f"  with phrase structure     : {stats['with_phrases']}")
        print(f"  usable for recommendation : {stats['usable']}")
        if stats["failed"]:
            print(f"  failed to parse           : {stats['failed']}")
    return stats


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    con = sqlite3.connect(db_path or paths.db_path())
    con.row_factory = sqlite3.Row
    return con


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else str(paths.db_path())
    print(f"Building {target} from {find_rekordbox()}")
    build(target)

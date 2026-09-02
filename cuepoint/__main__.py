"""Cuepoint's command line.

    cuepoint setup                      first run: scan, measure, then serve
    cuepoint doctor                     check ffmpeg, rekordbox, data folder

    python -m cuepoint build            rebuild the database from rekordbox
    python -m cuepoint extract          compute features for new tracks
    python -m cuepoint tracks [query]   search the library
    python -m cuepoint show <query>     one track's structure and key
    python -m cuepoint next <query>     what to play after this record
    python -m cuepoint set <query>      chain a whole set from one opener
    python -m cuepoint serve            the web UI, at localhost:8765
    python -m cuepoint index            build the audio-recognition index
    python -m cuepoint listen           live: hear what's playing, suggest next
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import library, mix

from . import paths

DB = None    # resolved by paths.db_path() at call time


# --------------------------------------------------------------------------
# presentation
# --------------------------------------------------------------------------

def clock(ms: int) -> str:
    s = max(0, ms) // 1000
    return f"{s // 60}:{s % 60:02d}"


def bar(score: float, width: int = 10) -> str:
    filled = int(round(score * width))
    return "#" * filled + "." * (width - filled)


def find(crate: dict, query: str) -> list:
    """Tracks whose title or filename contains `query`, case-insensitively."""
    q = query.lower().strip()
    hits = [t for t in crate.values()
            if q in t.title.lower() or q in t.filename.lower()]
    hits.sort(key=lambda t: (len(t.title), t.title))
    return hits


def resolve(crate: dict, query: str):
    """Pick the single track a query refers to, or explain why it can't."""
    if query.isdigit() and int(query) in crate:
        return crate[int(query)]
    hits = find(crate, query)
    if not hits:
        sys.exit(f"No track matches {query!r}. Try: python -m cuepoint tracks")
    if len(hits) > 1 and hits[0].title.lower() != query.lower().strip():
        print(f"{len(hits)} tracks match {query!r}; using the first:")
        for t in hits[:6]:
            print(f"   {t.id:>10}  {t.title[:58]}")
        print()
    return hits[0]


def describe(blend, index: int | None = None) -> None:
    """One recommendation, in the form a DJ would want to read it."""
    t = blend.track
    head = f"{index:>2}. " if index is not None else "    "
    ratio = ""
    if blend.ratio != 1.0:
        ratio = "  half-time" if blend.ratio == 2.0 else "  double-time"
    print(f"{head}{bar(blend.score)} {blend.score:.2f}  {t.title[:52]}")
    print(f"      {t.tempo:.1f} BPM  {t.key[0]:>4s} ({t.key[1]}){ratio}")
    print(f"      out at {clock(blend.cue_out_ms)} ({blend.from_phrase.role})"
          f"  ->  in at {clock(blend.cue_in_ms)} ({blend.to_phrase.role})"
          f"   {blend.bars} bars   pitch {blend.stretch * 100:+.1f}%")
    name, value = blend.weakest()
    detail = "  ".join(f"{k} {v:.2f}" for k, v in blend.parts.items())
    print(f"      {detail}")
    if value < 0.6:
        print(f"      watch the {name}")
    print()


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_build(args) -> None:
    print(f"Building {args.db} from {library.REKORDBOX}")
    library.build(args.db)
    print("\nNow run: python -m cuepoint extract")


def cmd_extract(args) -> None:
    from . import extract
    extract.run(args.db, force=args.force)


def cmd_tracks(args) -> None:
    con = library.connect(args.db)
    crate = mix.load_all(con)
    hits = find(crate, args.query) if args.query else sorted(
        crate.values(), key=lambda t: t.title)
    print(f"{len(hits)} of {len(crate)} mixable tracks\n")
    for t in hits[:args.limit]:
        print(f"  {t.id:>10}  {t.tempo:6.1f}  {t.key[1]:>3s}  {t.title[:56]}")
    if len(hits) > args.limit:
        print(f"\n  ... and {len(hits) - args.limit} more")


def cmd_show(args) -> None:
    con = library.connect(args.db)
    crate = mix.load_all(con)
    t = resolve(crate, args.query)
    print(f"{t.title}")
    print(f"{t.tempo:.1f} BPM   key {t.key[0]} ({t.key[1]})   "
          f"{clock(t.duration_ms)}   id {t.id}\n")
    print(f"  {'bar':>5} {'time':>6}  {'role':<10} {'bars':>4} "
          f"{'energy':>7} {'vocal':>6} {'kick':>6}")
    for p in t.phrases:
        print(f"  {p.start_bar:>5} {clock(p.start_ms):>6}  {p.role:<10} "
              f"{p.bars:>4} {p.energy_db:>6.1f}dB {p.vocal:>6.2f} "
              f"{p.kick_hz:>5.0f}Hz")


def cmd_next(args) -> None:
    con = library.connect(args.db)
    crate = mix.load_all(con)
    t = resolve(crate, args.query)
    print(f"Playing: {t.title}")
    print(f"         {t.tempo:.1f} BPM   key {t.key[0]} ({t.key[1]})\n")
    recs = mix.recommend(t, crate, limit=args.limit)
    if not recs:
        print("Nothing in the crate is within pitch range of this record.")
        return
    for i, blend in enumerate(recs, 1):
        describe(blend, i)


def cmd_set(args) -> None:
    """Chain recommendations into a full set, never repeating a record.

    Greedy: at each step take the best remaining blend. A greedy walk is the
    honest model of how the night actually goes -- you choose the next record
    knowing what is on the deck, not by planning two hours ahead.
    """
    con = library.connect(args.db)
    crate = mix.load_all(con)
    current = resolve(crate, args.query)

    played = {current.id}
    total = 0.0
    print(f"Set from: {current.title}\n")
    print(f"     1. {current.title[:56]}")
    print(f"        {current.tempo:.1f} BPM  {current.key[1]}  (opener)\n")

    for n in range(2, args.length + 1):
        recs = mix.recommend(current, crate, limit=1, exclude=played)
        if not recs:
            print("     (nothing left in range -- set ends here)")
            break
        blend = recs[0]
        total += blend.score
        print(f"  {n:>4}. {blend.track.title[:56]}")
        print(f"        {blend.track.tempo:.1f} BPM  {blend.track.key[1]}  "
              f"blend {blend.score:.2f} {bar(blend.score)}")
        print(f"        out {clock(blend.cue_out_ms)} "
              f"({blend.from_phrase.role}) -> in {clock(blend.cue_in_ms)} "
              f"({blend.to_phrase.role}), {blend.bars} bars, "
              f"pitch {blend.stretch * 100:+.1f}%\n")
        played.add(blend.track.id)
        current = blend.track

    if len(played) > 1:
        print(f"  {len(played)} tracks, mean blend "
              f"{total / (len(played) - 1):.2f}")


def cmd_index(args) -> None:
    from . import fingerprint
    fingerprint.build(args.db)


def cmd_listen(args) -> None:
    """Watch what the room is playing and keep a suggestion on screen."""
    from . import fingerprint, listen as listener_mod
    con = library.connect(args.db)
    crate = mix.load_all(con)
    try:
        index = fingerprint.Index()
    except FileNotFoundError:
        sys.exit("No audio index. Run: python -m cuepoint index")

    if args.list_devices:
        for i, name in listener_mod.devices():
            print(f"  [{i}] {name}")
        return

    lis = listener_mod.Listener(index, device=args.device).start()
    dev = listener_mod.devices()
    name = next((n for i, n in dev if i == lis.device), "?")
    print(f"Listening on [{lis.device}] {name}   (ctrl-c to stop)\n")

    last = None
    try:
        while True:
            time.sleep(1.0)
            s = lis.state
            if s.error:
                print(f"\r{s.error}", flush=True)
                break
            if s.track_id is None:
                print("\r  listening...".ljust(78), end="", flush=True)
                last = None
                continue
            t = crate.get(s.track_id)
            if t is None:
                continue
            if s.track_id != last:
                last = s.track_id
                print("\r".ljust(78))
                print(f"  NOW  {t.title[:56]}")
                print(f"       {t.tempo:.1f} BPM  {t.key[0]} ({t.key[1]})"
                      f"   confidence {s.confidence:.2f}")
                for i, b in enumerate(mix.recommend(t, crate, limit=3), 1):
                    print(f"    {i}. {b.score:.2f}  {b.track.title[:44]}"
                          f"   in at {clock(b.cue_in_ms)} ({b.to_role})")
                print()
            fix = "exact" if s.settled else "approx"
            print(f"\r       {clock(s.playhead_ms)} / {clock(t.duration_ms)}"
                  f"   [{fix}]   pitch {(s.rate - 1) * 100:+.1f}%".ljust(78),
                  end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        lis.stop()
        print("\nStopped.")


# --------------------------------------------------------------------------
# first run
# --------------------------------------------------------------------------

def _check_ffmpeg() -> tuple[bool, str]:
    from . import audio
    exe = shutil.which("ffmpeg") or audio.FFMPEG
    if not (exe and Path(exe).exists()):
        return False, "not found on PATH"
    try:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True,
                             timeout=15).stdout.splitlines()[0]
        return True, out[:60]
    except Exception as exc:                                 # noqa: BLE001
        return False, f"found but would not run ({exc})"


def cmd_doctor(args) -> None:
    """Everything Cuepoint needs, and whether this machine has it."""
    from . import paths
    ok = True

    print("Cuepoint check\n")

    have, detail = _check_ffmpeg()
    ok &= have
    print(f"  ffmpeg        {'OK  ' if have else 'MISSING'}  {detail}")
    if not have:
        print("                install it: brew install ffmpeg   (macOS)")
        print("                            winget install ffmpeg  (Windows)")

    rb = library.find_rekordbox()
    ok &= rb is not None
    print(f"  rekordbox     {'OK  ' if rb else 'MISSING'}  {rb or 'no collection found'}")
    if rb is None:
        for d in library.rekordbox_dirs():
            print(f"                looked in {d}")
        print("                override with CUEPOINT_REKORDBOX=/path/to/rekordbox")

    d = paths.data_dir()
    print(f"\n  data folder   {d}")
    db = paths.db_path()
    n_feat = len(list(paths.features_dir().glob('*.npz'))) if paths.features_dir().exists() else 0
    print(f"  database      {'present' if db.exists() else 'not built yet'}")
    print(f"  features      {n_feat} tracks measured")

    if db.exists():
        con = library.connect(args.db)
        total, = con.execute("SELECT COUNT(*) FROM tracks").fetchone()
        usable, = con.execute(
            "SELECT COUNT(*) FROM tracks WHERE analyzed=1 AND present=1").fetchone()
        con.close()
        print(f"  collection    {total} known, {usable} with audio + analysis")

    print("\n" + ("Ready. Run: cuepoint serve" if ok and db.exists()
                  else "Run: cuepoint setup" if ok else "Fix the items above first."))
    sys.exit(0 if ok else 1)


def cmd_setup(args) -> None:
    """Build, measure, and open the UI -- the whole first run, in order."""
    from . import paths
    have, detail = _check_ffmpeg()
    if not have:
        sys.exit(f"ffmpeg is required but was {detail}.\n"
                 "  macOS:   brew install ffmpeg\n"
                 "  Windows: winget install ffmpeg")
    if library.find_rekordbox() is None:
        sys.exit("No rekordbox collection found. Run `cuepoint doctor` for details.")

    paths.ensure()
    print(f"Data folder: {paths.data_dir()}\n")
    print("[1/3] Reading the rekordbox collection...")
    cmd_build(args)
    print("\n[2/3] Measuring each track (cached -- later runs only touch new ones)...")
    cmd_extract(args)
    print("\n[3/3] Starting the web UI...")
    cmd_serve(args)


def cmd_serve(args) -> None:
    from . import server
    server.serve(args.db, args.port)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="cuepoint", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB, help="database path")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="rebuild from rekordbox").set_defaults(
        func=cmd_build)

    p = sub.add_parser("extract", help="compute missing features")
    p.add_argument("--force", action="store_true", help="redo every track")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("tracks", help="search the library")
    p.add_argument("query", nargs="?", default="")
    p.add_argument("-n", "--limit", type=int, default=40)
    p.set_defaults(func=cmd_tracks)

    p = sub.add_parser("show", help="one track's structure")
    p.add_argument("query")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("next", help="what to play next")
    p.add_argument("query")
    p.add_argument("-n", "--limit", type=int, default=8)
    p.set_defaults(func=cmd_next)

    p = sub.add_parser("set", help="chain a whole set")
    p.add_argument("query")
    p.add_argument("-n", "--length", type=int, default=10)
    p.set_defaults(func=cmd_set)

    p = sub.add_parser("index", help="build the audio-recognition index")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("listen", help="live recognition + suggestions")
    p.add_argument("-d", "--device", type=int, default=None,
                   help="audio input index (default: loopback if present)")
    p.add_argument("--list-devices", action="store_true")
    p.set_defaults(func=cmd_listen)

    sub.add_parser("doctor", help="check ffmpeg, rekordbox and the data folder"
                   ).set_defaults(func=cmd_doctor)

    p = sub.add_parser("setup", help="first run: build, measure, then serve")
    p.add_argument("--force", action="store_true", help="redo every track")
    p.add_argument("-p", "--port", type=int, default=8765)
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("serve", help="run the web UI")
    p.add_argument("-p", "--port", type=int, default=8765)
    p.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

# Cuepoint

Cuepoint answers one question: **this record is playing — what comes next, and
where exactly do I bring it in?**

Every other tool stops at tempo and key. Two records can share both and still
wreck: the kicks land on different sixteenths, or both singers arrive at once.
Cuepoint listens to the parts that actually decide it.

It reads a local rekordbox collection strictly read-only. It never writes into
rekordbox's own files.

## Why it can do this

rekordbox has already analysed the collection — beat grid, downbeats, and the
phrase structure (intro / build / peak / breakdown / outro). That analysis sits
in the clear in `networkAnalyze6.db` and the `.DAT`/`.EXT` files beside it, so
none of it has to be recomputed. Cuepoint parses it, then decodes each track
once to measure what the analysis does not carry:

| feature | what it settles |
|---|---|
| `groove_low` | where the kick sits in the bar — the clash you can't hear coming |
| `groove_full` | whether the overall feel matches, swung against straight |
| `chroma` | harmonic agreement, from the actual spectrum rather than a key label |
| `vocal` | is somebody singing here |
| `energy_db` | does the incoming record hold the level |
| `kick_hz` | do the two kick fundamentals beat against each other |

Grooves are measured against rekordbox's own grid and expressed as 24 slots of
one bar, so two records at different tempos compare directly.

## How a blend is scored

A recommendation is never just a track — it is a phrase of the outgoing record,
a phrase of the incoming one, and the number of bars they run together. Seven
components, each 0..1:

    tempo 0.15   can the pitch fader reach it (also half and double time)
    groove 0.22  do the two bars agree about where the weight falls
    vocal 0.18   two vocals at once, the most audible mistake there is
    harmonic 0.15   do the pitch classes agree
    energy 0.12  does the level hold or lift
    role 0.13    is this a sensible place to leave, and to arrive
    sub 0.05     do the kick fundamentals fight in the bottom octave

Thresholds are measured, not guessed. Two phrases picked at random correlate at
0.14 in the low band while two phrases of the *same* record correlate at 0.81,
and the ramps are set across gaps like that one. Where a refinement failed to
earn its place it was removed and the reason left in the source — see
`harmonic_fit` in [`mix.py`](cuepoint/mix.py).

## Install

Requires **Python 3.9+**, **ffmpeg** on `PATH`, and a **rekordbox** collection
that has been analysed. numpy and scipy come in with the package.

    pip install cuepoint      # or: pipx install cuepoint
    cuepoint doctor           # checks ffmpeg, rekordbox, and where data lives
    cuepoint setup            # scan, measure, then open the UI

`setup` is the whole first run. It reads the rekordbox collection, decodes each
track once to measure what the analysis does not carry, and opens the web UI on
localhost:8765. Expect a few minutes for a few hundred tracks; the measurement
is cached per track, so later runs only touch what changed.

Everything Cuepoint derives lives in one folder, away from wherever you happen
to be standing:

    macOS     ~/Library/Application Support/Cuepoint
    Windows   %LOCALAPPDATA%\Cuepoint
    Linux     ~/.local/share/cuepoint

Set `CUEPOINT_HOME` to move it, or `CUEPOINT_REKORDBOX` if your collection is
not where rekordbox usually puts it. A directory that already contains a
`cuepoint.db` keeps using it, so nothing is stranded by upgrading.

## Use

After the first run, the individual steps are there when you need them:

    cuepoint build      # re-read rekordbox (after moving or adding music)
    cuepoint extract    # measure new tracks
    cuepoint serve      # the web UI

Run those three in that order whenever you move files or add music -- each one
feeds the next.

From the command line:

    cuepoint tracks eleven          # search
    cuepoint show "Ten to Eleven"   # structure, key, per-phrase detail
    cuepoint next "Ten to Eleven"   # what to play next
    cuepoint set  "Ten to Eleven" -n 10   # chain a whole set

`next` prints the mix points and the full score breakdown:

    1. #########. 0.92  Find Your Corner
          126.0 BPM     C (8B)
          out at 2:17 (peak)  ->  in at 0:30 (build)   4 bars   pitch +0.0%
          tempo 1.00  groove 1.00  vocal 0.98  harmonic 1.00  energy 0.99 ...

The web UI draws the same thing: both records on one time axis, aligned at the
bar where one becomes the other, plus a compatible-track filter -- pitch range
and Camelot key, the way a DJ browser works.

## Live mode

Not currently wired in. `fingerprint.py` and `listen.py` still build a
recognition index and follow the room, and `cuepoint index` / `cuepoint listen`
still run -- but the live capture path degrades chroma badly enough that the
matcher will not commit, so it is out of the UI until that is fixed. A track
matched from its file scores 0.998; the same audio captured live scores 0.94
with almost no margin over the runner-up.

## Layout

    anlz.py       parser for rekordbox's .DAT/.EXT analysis files
    library.py    builds cuepoint.db from the collection (read-only)
    audio.py      ffmpeg decode, spectrogram, onset envelope, grid alignment
    features.py   per-phrase groove / chroma / vocal / energy
    extract.py    runs features.py across the library, in parallel, cached
    mix.py        the scorer: what follows what, and where
    fingerprint.py  chroma index + closed-set recognition and playhead search
    listen.py     audio capture, hypothesis tracking, live state
    server.py     local web UI
    paths.py      where the database, features and index live, per platform
    static/       the UI itself, and the two typefaces it sets (SIL OFL 1.1),
                  self-hosted so a booth with no wifi still looks right

Deliberately no librosa — everything used here is a few dozen lines of numpy.

## Caveats

`vocal` is a heuristic (harmonic/percussive separation plus a band-limited
energy ratio), not a trained detector. It separates "instrumental" from
"someone is singing", which is what the scorer needs; it will not catch a
heavily processed vocal chop.

Phrase roles come from rekordbox and inherit its mistakes — occasional two-bar
fragments are segmentation noise, and are skipped as mix points.

Live recognition assumes keylock is on, which is the modern default. With
keylock off, a large pitch move shifts energy across chroma bins and match
confidence will fall.

Streaming catalogues (Beatport, Beatsource, TIDAL) are not integrated. The
Beatport v4 API is OAuth behind an approved-partner portal and in-app streaming
needs a commercial LINK agreement, so it is not reachable without a deal.

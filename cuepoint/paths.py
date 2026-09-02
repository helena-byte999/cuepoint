"""Where Cuepoint keeps the things it derives.

A tool that people download cannot scatter a database, a feature cache and a
20 MB index across whatever directory they happened to be standing in when
they ran it. Everything Cuepoint computes lives in one per-user directory.

Two escape hatches, in order of precedence:

    CUEPOINT_HOME=/some/path     explicit, wins over everything
    ./cuepoint.db present        a working tree that already has data keeps
                                 using it, so upgrading never strands anyone

Nothing here writes: these are locations, not side effects. `ensure()` is the
only function that creates anything, and callers ask for it explicitly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DB_NAME = "cuepoint.db"
INDEX_NAME = "fingerprint.npz"
FEATURES_NAME = "features"


def _platform_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/Cuepoint"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData/Local"
        return Path(base) / "Cuepoint"
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share"
    return Path(base) / "cuepoint"


def data_dir() -> Path:
    """The directory holding this user's database, features and index."""
    env = os.environ.get("CUEPOINT_HOME")
    if env:
        return Path(env).expanduser()
    # An existing working tree wins, so a developer -- or anyone who ran the
    # older cwd-relative version -- does not silently lose their extraction.
    if Path(DB_NAME).exists():
        return Path.cwd()
    return _platform_dir()


def ensure() -> Path:
    d = data_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / FEATURES_NAME).mkdir(exist_ok=True)
    return d


def db_path() -> Path:
    return data_dir() / DB_NAME


def features_dir() -> Path:
    return data_dir() / FEATURES_NAME


def index_path() -> Path:
    return data_dir() / INDEX_NAME

"""A local web UI for Cuepoint.

Reading a transition as text works, but a blend is a fundamentally visual
thing: two records laid side by side, aligned at the bar where one becomes the
other. That is what this serves.

Standard library only -- no framework. The crate is loaded once at startup and
held in memory (a few hundred tracks is a few megabytes), so every query is
instant. Binds to localhost: this reads a personal music library and is not
intended to be reachable from anywhere else.

Live recognition is not wired in here for now -- listen.py and fingerprint.py
are still in the tree, but the capture path needs work before it earns a place
in the UI.
"""

from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from . import library, mix

STATIC = Path(__file__).parent / "static"


def phrase_json(p) -> dict:
    return {"idx": p.idx, "role": p.role, "start_ms": p.start_ms,
            "end_ms": p.end_ms, "bars": p.bars, "start_bar": p.start_bar,
            "energy_db": round(p.energy_db, 1), "vocal": round(p.vocal, 3),
            "kick_hz": round(p.kick_hz, 1)}


def track_json(t, phrases: bool = False) -> dict:
    key, camelot = t.key
    out = {"id": t.id, "title": t.title, "tempo": round(t.tempo, 1),
           "key": key, "camelot": camelot, "duration_ms": t.duration_ms,
           "folder": t.folder}
    if phrases:
        out["phrases"] = [phrase_json(p) for p in t.phrases]
    return out


def blend_json(b) -> dict:
    name, value = b.weakest()
    return {
        "track": track_json(b.track, phrases=True),
        "score": round(b.score, 4),
        "parts": {k: round(v, 4) for k, v in b.parts.items()},
        "cue_out_ms": b.cue_out_ms, "cue_in_ms": b.cue_in_ms,
        "from_role": b.from_phrase.role, "to_role": b.to_phrase.role,
        "from_idx": b.from_phrase.idx, "to_idx": b.to_phrase.idx,
        "bars": b.bars, "ratio": b.ratio,
        "stretch_pct": round(b.stretch * 100, 1),
        "weakest": name, "weakest_value": round(value, 3),
    }


class Handler(SimpleHTTPRequestHandler):
    crate: dict = {}
    # Stated outright rather than left to the host's mime.types, which does
    # not carry woff2 everywhere. The UI ships its own typefaces.
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".woff2": "font/woff2"}

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(STATIC), **kw)

    def log_message(self, fmt, *args):      # quiet; this is a local tool
        pass

    def _send(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if not url.path.startswith("/api/"):
            return super().do_GET()
        try:
            self._api(url.path[5:], parse_qs(url.query))
        except Exception as exc:                     # noqa: BLE001
            self._send({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _api(self, route: str, q: dict) -> None:
        crate = Handler.crate
        if route == "tracks":
            query = (q.get("q") or [""])[0].lower().strip()
            hits = [t for t in crate.values()
                    if not query or query in t.title.lower()]
            hits.sort(key=lambda t: t.title.lower())
            return self._send({"tracks": [track_json(t) for t in hits]})

        if route.startswith("track/"):
            t = crate.get(int(route.split("/")[1]))
            if t is None:
                return self._send({"error": "no such track"}, 404)
            return self._send(track_json(t, phrases=True))

        if route.startswith("next/"):
            t = crate.get(int(route.split("/")[1]))
            if t is None:
                return self._send({"error": "no such track"}, 404)
            n = int((q.get("n") or ["12"])[0])
            recs = mix.recommend(t, crate, limit=n)
            return self._send({"playing": track_json(t, phrases=True),
                               "blends": [blend_json(b) for b in recs]})

        self._send({"error": "unknown route"}, 404)


def serve(db_path: str = "cuepoint.db", port: int = 8765) -> None:
    # flush=True throughout: stdout is block-buffered whenever this is piped to
    # a log rather than a terminal, and a server that looks silent while it is
    # actually serving is worse than no logging at all.
    con = library.connect(db_path)
    print("Loading crate...", flush=True)
    Handler.crate = mix.load_all(con)
    con.close()
    print(f"{len(Handler.crate)} tracks ready", flush=True)

    print(f"Cuepoint -> http://127.0.0.1:{port}", flush=True)
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)


if __name__ == "__main__":
    serve()

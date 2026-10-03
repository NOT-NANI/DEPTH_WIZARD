#!/usr/bin/env python3
"""Unified DepthWizard entrypoint for live deployment (Hugging Face Spaces / Docker / Cloud)."""
from __future__ import annotations

import mimetypes
import os
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.run_demo import Handler as BaseHandler, RUNS

FRONTEND_DIR = ROOT / "frontend"


class LiveHandler(BaseHandler):
    """Subclass of run_demo Handler that serves the modern frontend at / and static files."""

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        request = urlsplit(self.path)
        clean_path = request.path.strip("/")

        # 1. API routes and artifact files handled by the original backend Handler
        if request.path == "/api/health" or request.path.startswith("/api/") or request.path.startswith("/files/"):
            super().do_GET()
            return

        # 2. Serve index.html for root
        if request.path == "/" or request.path == "/index.html":
            index_file = FRONTEND_DIR / "index.html"
            if index_file.is_file():
                self._send(200, index_file.read_bytes(), "text/html; charset=utf-8")
                return

        # 3. Serve static assets from frontend/ (css, js, vendor, etc.)
        target = (FRONTEND_DIR / clean_path).resolve()
        try:
            target.relative_to(FRONTEND_DIR.resolve())
        except ValueError:
            self._send(403, b"Forbidden", "text/plain")
            return

        if target.is_file():
            ctype, _ = mimetypes.guess_type(target.name)
            if target.suffix == ".js":
                ctype = "application/javascript; charset=utf-8"
            elif target.suffix == ".css":
                ctype = "text/css; charset=utf-8"
            elif target.suffix == ".html":
                ctype = "text/html; charset=utf-8"
            elif not ctype:
                ctype = "application/octet-stream"
            self._send(200, target.read_bytes(), ctype)
            return

        # 4. Fallback to base handler
        super().do_GET()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    port_env = int(os.environ.get("PORT", "7860"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=port_env)
    args = parser.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), LiveHandler)
    print(f"DepthWizard Live running on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

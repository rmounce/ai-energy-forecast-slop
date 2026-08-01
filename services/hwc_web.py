#!/usr/bin/env python3
"""Read-only HTTP API and static server for the HWC efficiency dashboard.

The HWC daemon remains the sole writer of the SQLite store.  This process uses aiohttp for
the small async web surface and opens a fresh read-only SQLite connection per request.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sqlite3
import time
from pathlib import Path

from aiohttp import web


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data/hwc_cycles.sqlite"
DEFAULT_WEB_ROOT = REPO_ROOT / "web/hwc"
MAX_CYCLES = 10_000

log = logging.getLogger("hwc_web")


def _json_value(value):
    """Convert SQLite values to strict JSON values."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _row_dict(row: sqlite3.Row) -> dict:
    return {key: _json_value(row[key]) for key in row.keys()}


def _read_db(db_path: Path, query: str, params=()) -> list[dict]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [_row_dict(row) for row in conn.execute(query, params)]
    finally:
        conn.close()


def _db_path(request: web.Request) -> Path:
    return request.app["db_path"]


async def health(_request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "service": "hwc_web", "time": time.time()})


async def cycles(request: web.Request) -> web.Response:
    try:
        limit = min(int(request.query.get("limit", MAX_CYCLES)), MAX_CYCLES)
    except ValueError:
        raise web.HTTPBadRequest(text="limit must be an integer")
    rows = await asyncio.to_thread(
        _read_db,
        _db_path(request),
        "SELECT * FROM hwc_cycles WHERE status = 'complete' ORDER BY start_ts LIMIT ?",
        (limit,),
    )
    return web.json_response({"cycles": rows, "generated_at": time.time()})


async def trace(request: web.Request) -> web.Response:
    try:
        start_ts = float(request.match_info["start_ts"])
    except ValueError:
        raise web.HTTPBadRequest(text="cycle id must be a start timestamp")
    rows = await asyncio.to_thread(
        _read_db,
        _db_path(request),
        "SELECT * FROM hwc_cycle_samples WHERE cycle_start_ts = ? ORDER BY ts",
        (start_ts,),
    )
    return web.json_response({"cycle_start_ts": start_ts, "samples": rows})


async def index(request: web.Request) -> web.StreamResponse:
    return web.FileResponse(request.app["web_root"] / "index.html")


def create_app(db_path: Path, web_root: Path) -> web.Application:
    app = web.Application()
    app["db_path"] = db_path.resolve()
    app["web_root"] = web_root.resolve()
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/cycles", cycles)
    app.router.add_get("/api/cycles/{start_ts}/trace", trace)
    app.router.add_get("/", index)
    app.router.add_static("/", app["web_root"], show_index=False)
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the HWC efficiency dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--web-root", type=Path, default=DEFAULT_WEB_ROOT)
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error(f"SQLite database does not exist: {args.db}")
    if not (args.web_root / "index.html").is_file():
        parser.error(f"dashboard index does not exist: {args.web_root / 'index.html'}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.info("serving %s on http://%s:%d", args.web_root, args.host, args.port)
    web.run_app(create_app(args.db, args.web_root), host=args.host, port=args.port, access_log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

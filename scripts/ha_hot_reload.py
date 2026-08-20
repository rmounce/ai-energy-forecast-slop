#!/usr/bin/env python3
"""Hot-reload HA templates and save a storage-mode Lovelace dashboard."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_utils import load_config


def load_ha_config(path: Path) -> tuple[str, str]:
    config = load_config(path)
    ha = config["home_assistant"]
    token = os.environ.get("HA_TOKEN") or str(ha.get("token") or "")
    if not token:
        raise RuntimeError(
            "No Home Assistant token: set HA_TOKEN or home_assistant.token in config.yaml"
        )
    return str(ha["url"]).rstrip("/"), token


def reload_templates(base_url: str, token: str) -> None:
    response = requests.post(
        f"{base_url}/api/services/template/reload",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    response.raise_for_status()


def websocket_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, "/api/websocket", "", "", ""))


async def ws_command(socket, message_id: int, **payload):
    await socket.send(json.dumps({"id": message_id, **payload}))
    while True:
        response = json.loads(await socket.recv())
        if response.get("id") != message_id:
            continue
        if not response.get("success"):
            raise RuntimeError(f"Home Assistant WebSocket command failed: {response.get('error')}")
        return response.get("result")


async def save_lovelace(
    base_url: str,
    token: str,
    url_path: str,
    storage_path: Path,
    backup_path: Path,
) -> None:
    with storage_path.open() as handle:
        stored = json.load(handle)
    desired = stored["data"]["config"]

    async with websockets.connect(websocket_url(base_url), open_timeout=30) as socket:
        greeting = json.loads(await socket.recv())
        if greeting.get("type") != "auth_required":
            raise RuntimeError(f"Unexpected Home Assistant greeting: {greeting.get('type')}")
        await socket.send(json.dumps({"type": "auth", "access_token": token}))
        auth = json.loads(await socket.recv())
        if auth.get("type") != "auth_ok":
            raise RuntimeError(f"Home Assistant authentication failed: {auth.get('message')}")

        current = await ws_command(
            socket, 1, type="lovelace/config", url_path=url_path, force=True
        )
        backup_path.write_text(json.dumps(current, indent=2) + "\n")
        await ws_command(
            socket,
            2,
            type="lovelace/config/save",
            url_path=url_path,
            config=desired,
        )
        saved = await ws_command(
            socket, 3, type="lovelace/config", url_path=url_path, force=True
        )
        if saved != desired:
            raise RuntimeError("Dashboard verification failed after save")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--reload-templates", action="store_true")
    parser.add_argument("--lovelace-storage", type=Path)
    parser.add_argument("--dashboard-url-path")
    parser.add_argument(
        "--dashboard-backup",
        type=Path,
        default=Path("/tmp/ha-lovelace-live-backup.json"),
    )
    args = parser.parse_args()
    if args.lovelace_storage and not args.dashboard_url_path:
        parser.error("--lovelace-storage requires --dashboard-url-path")
    if not args.reload_templates and not args.lovelace_storage:
        parser.error("select --reload-templates and/or --lovelace-storage")
    return args


def main() -> None:
    args = parse_args()
    base_url, token = load_ha_config(args.config)
    if args.reload_templates:
        reload_templates(base_url, token)
        print("Reloaded Home Assistant template entities")
    if args.lovelace_storage:
        asyncio.run(
            save_lovelace(
                base_url,
                token,
                args.dashboard_url_path,
                args.lovelace_storage,
                args.dashboard_backup,
            )
        )
        print(
            f"Saved dashboard {args.dashboard_url_path}; "
            f"previous live config: {args.dashboard_backup}"
        )


if __name__ == "__main__":
    main()

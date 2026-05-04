import json
from pathlib import Path


def load_config(path="config.json"):
    """Load config.json and deep-merge config.secrets.json from the same directory if present."""
    path = Path(path)
    with open(path) as f:
        config = json.load(f)
    secrets_path = path.parent / "config.secrets.json"
    if secrets_path.exists():
        with open(secrets_path) as f:
            _deep_merge(config, json.load(f))
    return config


def _deep_merge(base, override):
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value

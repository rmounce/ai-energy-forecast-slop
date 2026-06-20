from pathlib import Path

import yaml


def load_config(path="config.yaml"):
    """Load config.yaml and deep-merge config.secrets.yaml from the same directory if present."""
    path = Path(path)
    with open(path) as f:
        config = yaml.safe_load(f)
    secrets_path = path.parent / "config.secrets.yaml"
    if secrets_path.exists():
        with open(secrets_path) as f:
            _deep_merge(config, yaml.safe_load(f))
    return config


def _deep_merge(base, override):
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value

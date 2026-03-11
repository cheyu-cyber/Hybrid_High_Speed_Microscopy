"""Load configuration from config.json."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


def load_config(section: str) -> SimpleNamespace:
    """Load a section from config.json and return it as a namespace with attribute access."""
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if section not in data:
        raise KeyError(f"Section '{section}' not found in {CONFIG_PATH}")
    return SimpleNamespace(**data[section])

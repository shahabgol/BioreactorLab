"""Process feed composition, with independent synthetic defaults."""

from math import isfinite
from os import environ
from pathlib import Path

import yaml

_DEMO = {"A": {"GLC": 400.0}, "B": {"GLC": 50.0, "GLU": 2.0, "GLN": 0.02}}


def load_feed_concentrations(path):
    """Read the supported Feed A/B solutes in g/L from a local YAML file."""
    recipes = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["feeds_g_per_L"]
    result = {
        feed: {name: float(recipes[feed][name]) for name in solutes}
        for feed, solutes in _DEMO.items()
    }
    if any(not isfinite(value) or value < 0 for feed in result.values() for value in feed.values()):
        raise ValueError("Feed concentrations must be finite, nonnegative values in g/L.")
    return result


def active_feed_concentrations():
    configured = environ.get("BIOREACTOR_FEED_CONFIG")
    if configured:
        return load_feed_concentrations(configured)
    local = Path(__file__).resolve().parents[3] / "configs" / "development.yaml"
    return load_feed_concentrations(local) if local.is_file() else _DEMO


FEED_CONCENTRATIONS = active_feed_concentrations()

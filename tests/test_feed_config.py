"""Ensure licensed public defaults and private feed overrides remain separate."""

import pytest
from bioreactor_twin.data import feeds


def test_feed_selection_prefers_explicit_then_local_then_demo(tmp_path, monkeypatch):
    monkeypatch.delenv("BIOREACTOR_FEED_CONFIG", raising=False)
    monkeypatch.setattr(feeds, "__file__", str(tmp_path / "main/bioreactor_twin/data/feeds.py"))
    demo = feeds.active_feed_concentrations()
    local = tmp_path / "configs/development.yaml"
    local.parent.mkdir()
    local.write_text("feeds_g_per_L: {A: {GLC: 300}, B: {GLC: 30, GLU: 1, GLN: 0.03}}\n")
    assert feeds.active_feed_concentrations()["A"]["GLC"] == 300
    explicit = tmp_path / "chosen.yaml"
    explicit.write_text("feeds_g_per_L: {A: {GLC: 200}, B: {GLC: 20, GLU: 1, GLN: 0}}\n")
    monkeypatch.setenv("BIOREACTOR_FEED_CONFIG", str(explicit))
    assert feeds.active_feed_concentrations()["A"]["GLC"] == 200
    explicit.write_text("feeds_g_per_L: {A: {GLC: -1}, B: {GLC: 20, GLU: 1, GLN: 0}}\n")
    with pytest.raises(ValueError, match="nonnegative"):
        feeds.active_feed_concentrations()
    assert demo["A"]["GLC"] > 0

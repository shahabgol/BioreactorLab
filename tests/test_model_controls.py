"""User-defined grouping must reach every validation layer, not only the UI."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from bioreactor_twin.calibration.validation import (
    condition_groups,
    linked_test_rows,
    validation_splits,
)
from bioreactor_twin.data.dataset import condition_key, condition_size
from bioreactor_twin.hybrid.training import fingerprint, train_hybrid
from bioreactor_twin.models.kinetics import Parameters
from bioreactor_twin.web.app import create_app
from bioreactor_twin.web.equations import equations
from bioreactor_twin.web.workspace import Lab, Workspace
from dash import no_update
from test_hybrid import data


def test_custom_groups_and_checkbox_linking():
    runs = {
        n: replace(data(n), replicate_group="pair" if n in (7, 8) else str(n)) for n in (1, 2, 7, 8)
    }
    assert condition_groups(runs) == [(1,), (2,), (7, 8)]
    assert condition_size(runs[7], runs.values()) == 2
    assert condition_key(runs[1]) == "1"
    assert linked_test_rows(list(runs), [2], [], runs) == [2, 3]
    assert linked_test_rows(list(runs), [3], [2, 3], runs) == []
    assert validation_splits(runs, "train_test", [7, 8])[0].test == (7, 8)
    with pytest.raises(ValueError, match="Replicates"):
        validation_splits(runs, "train_test", [7])
    assert [s.test for s in validation_splits(runs, "cross_validation")] == [(1,), (2,), (7, 8)]
    assert fingerprint(runs) != fingerprint({n: data(n) for n in runs})


def test_group_edit_ui_feed_conservation_and_equation_coverage(tmp_path, monkeypatch):
    import bioreactor_twin.web.callbacks as callbacks

    lab = Lab(tmp_path)
    lab.sessions["x"] = Workspace({n: data(n) for n in (1, 2, 7, 8)}, "synthetic", Parameters())
    cb = create_app(lab=lab).twin_callbacks
    monkeypatch.setattr(callbacks, "ctx", SimpleNamespace(triggered_id="mark-replicates"))
    rows = cb["explore"]([1], "VCD", 0, "feeds", "x")[2]
    figure = cb["explore"]([1, 7], "VCD", 0, "feeds", "x")[0]
    assert len(figure.data) == 6
    assert sum(figure.data[1].y) == pytest.approx(lab.get("x").runs[1].events[:, 2].sum() * 1000)
    assert cb["set_replicates"](1, 0, [], rows, "x", 0)[0] is no_update
    assert cb["set_replicates"](1, 0, [2, 3], rows, "x", 0)[0] == 1
    work = lab.get("x")
    assert work.parameters is None
    plan = cb["validation_plan"]("train_test", [2], 1, "x", dict(revision=1, selected=[]))
    assert plan[1]["test_runs"] == [7, 8]
    cv = cb["validation_plan"]("cross_validation", [], 1, "x", dict(revision=1, selected=[]))
    assert cv[3][-1]["test"] == "7, 8"
    assert cb["populate"](1, "x")[-1][-1]["group"] == "Replicates 7, 8"
    monkeypatch.setattr(callbacks, "ctx", SimpleNamespace(triggered_id="ungroup-replicates"))
    assert cb["set_replicates"](0, 1, [2], rows, "x", 1)[0] == 2
    assert condition_key(lab.get("x").runs[7]) == "Condition 7"
    lab.get("x").job = {"status": "running"}
    with pytest.raises(ValueError, match="Wait"):
        lab.set_replicates("x", [1, 2])
    text = equations(False).children
    assert all(name in text for name in asdict(Parameters()))
    assert "inactive" in cb["equations"]("hybrid").children
    assert "lactate_yield" in cb["equations"]("first_principles").children
    lab.executor.shutdown(wait=True)


def test_nested_hybrid_respects_custom_groups(monkeypatch):
    import bioreactor_twin.hybrid.training as module

    seen = []

    def fit(runs, *args, **kwargs):
        ids = {r.number for r in runs}
        assert (7 in ids) == (8 in ids)
        seen.append(ids)
        return Parameters(), {"success": True}

    monkeypatch.setattr(module, "fit_core", fit)
    runs = {
        n: replace(data(n), replicate_group="pair" if n in (7, 8) else str(n)) for n in (1, 2, 7, 8)
    }
    result = train_hybrid(runs, "cross_validation", options={"family": "constant"})
    assert result["validation"]["folds"][-1]["test_runs"] == [7, 8]
    assert seen

"""Callback integration and HTTP delivery tests using the real Dash app."""

import base64
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from bioreactor_twin.models.kinetics import Parameters
from bioreactor_twin.simulation.nitrogen import Nitrogen
from bioreactor_twin.web.app import create_app
from bioreactor_twin.web.charts import comparison, exploration, replicate_figure, simulation_figure
from bioreactor_twin.web.services import (
    evaluate,
    replicate_analysis,
    scenario_defaults,
)
from bioreactor_twin.web.workspace import Lab, Workspace
from dash import no_update
from test_io_workflow import workbook
from test_numerics import synthetic_run


@pytest.fixture
def application(tmp_path):
    lab = Lab(tmp_path)
    runs = {n: synthetic_run(n) for n in (1, 2, 7, 8)}
    work = Workspace(runs, "synthetic.xlsx", Parameters(), (7, 8), "Baseline")
    work.predictions, work.scores = evaluate(runs, work.parameters)
    lab.sessions["test"] = work
    app = create_app(lab=lab)
    yield app
    lab.executor.shutdown(wait=True)


def trigger(monkeypatch, identity):
    import bioreactor_twin.web.callbacks as module

    monkeypatch.setattr(module, "ctx", SimpleNamespace(triggered_id=identity))


def test_real_http_layout_assets_and_dash_callback(application):
    client = application.server.test_client()
    for path in ("/", "/_dash-layout", "/_dash-dependencies", "/assets/lab.css"):
        assert client.get(path).status_code == 200
    loader = client.get("/assets/bioreactor-loader.svg")
    assert loader.status_code == 200 and b"@keyframes spin" in loader.data
    layout = client.get("/_dash-layout").get_json()
    assert layout["props"]["target_components"]["task2-result"] == "data"
    assert "Simulation" in str(layout) and "Model training" in str(layout)
    key = next(k for k in application.callback_map if "page-title" in k)
    outputs = [
        {"id": f"controls-{name}", "property": "style"}
        for name in ("explore", "replicates", "training", "simulation")
    ]
    outputs += [
        {"id": f"results-{name}", "property": "style"}
        for name in ("explore", "replicates", "training", "simulation")
    ]
    outputs += [
        {"id": name, "property": "children"}
        for name in ("page-eyebrow", "page-title", "page-subtitle")
    ]
    response = client.post(
        "/_dash-update-component",
        json={
            "output": key,
            "outputs": outputs,
            "inputs": [{"id": "navigation", "property": "value", "value": "simulation"}],
            "state": [],
            "changedPropIds": ["navigation.value"],
        },
    )
    assert response.status_code == 200
    assert response.get_json()["response"]["page-title"]["children"] == "Operating simulations"


def test_population_exploration_and_model_views(application):
    cb = application.twin_callbacks
    assert cb["navigate"]("training")[-2] == "Model training"
    assert cb["populate"](0, "test")[1] == [1, 2, 7, 8]
    assert cb["populate"](0, "empty")[1] == []
    result = cb["explore"]([1, 2], "GLC", 0, "assays", "test")
    assert len(result[0].data) == 2 and len(result[1]) == 10
    assert cb["explore"]([], "VCD", 0, "assays", "test")[1] == []
    assert len(cb["training_results"](0, 0, "VCD", [1, 2], "test")[1]) == 14
    assert cb["training_results"](0, 0, "VCD", [], "empty")[1] == []
    work = application.lab.get("test")
    work.trained = (1, 2, 7, 8)
    assert "None" in str(cb["training_results"](0, 0, "GLC", [1], "test")[3])


def test_import_callbacks_success_and_preserving_previous_data(application, tmp_path, monkeypatch):
    cb = application.twin_callbacks["import"]
    trigger(monkeypatch, "upload")
    path = workbook(tmp_path)
    contents = "data:x;base64," + base64.b64encode(path.read_bytes()).decode()
    assert cb(contents, 0, "new.xlsx", "test", 0)[0] == 1
    assert cb("invalid", 0, "bad.csv", "test", 1)[0] is no_update
    assert list(application.lab.get("test").runs) == [7]
    trigger(monkeypatch, "example")
    assert cb(None, 1, None, "test", 1)[0] == 2


def test_replicate_views_and_error_messages(application):
    cb = application.twin_callbacks["replicates"]
    assert len(cb(0, "VCD", 0, 1, 2, 0, "test")[1]) == 5
    assert cb(0, "VCD", 0, 1, 1, 0, "test")[1] == []


def test_validation_table_linking_and_cross_validation_plan(application):
    cb = application.twin_callbacks["validation_plan"]
    first = cb("train_test", [], 0, "test", None)
    assert first[0] == [0, 1] and first[1]["valid"]
    assert first[3][0]["training"] == "7, 8"
    removed = cb("train_test", [1], 0, "test", first[1])
    assert removed[0] == []
    added = cb("train_test", [0], 0, "test", removed[1])
    assert added[0] == [0, 1]
    cv = cb("cross_validation", [0, 1], 0, "test", added[1])
    assert [r["test"] for r in cv[3]] == ["1, 2", "7", "8"]
    assert cv[4] == {"display": "none"} and cv[5] == {}
    invalid = cb("train_test", [0, 1, 2, 3], 0, "test", added[1])
    assert not invalid[1]["valid"] and "training" in invalid[2]
    assert not cb("train_test", [], 0, "empty", None)[1]["valid"]
    note = application.twin_callbacks["model_type_note"]
    assert "learned" in note("hybrid")
    assert "Monod" in note("first_principles")


def test_cv_view_labels_and_linked_legend(application):
    work = application.lab.get("test")
    work.validation = {"mode": "cross_validation", "folds": [{}, {}, {}]}
    result = application.twin_callbacks["training_results"](1, 0, "VCD", [1, 2], "test")
    figure = result[0]
    assert len(figure.data) == 4
    assert figure.data[0].legendgroup == figure.data[1].legendgroup == "run-1"
    assert figure.layout.legend.groupclick == "togglegroup"
    assert "CV held-out" in figure.data[0].name
    assert all(r["split"] == "CV held-out" for r in result[2])
    assert "out-of-fold" in result[4]
    assert len(application.twin_callbacks["training_results"](1, 0, "VCD", [], "test")[0].data) == 0


def test_training_controls_and_progress_states(application, monkeypatch):
    cb = application.twin_callbacks
    assert "Review" in cb["train"](1, 1, 20, "train_test", "first_principles", None, "test")
    plan = {"valid": True, "mode": "train_test", "test_runs": [1, 2]}
    monkeypatch.setattr(application.lab, "train", lambda *args, **kwargs: "Training queued")
    assert (
        cb["train"](1, 1, 20, "train_test", "first_principles", plan, "test") == "Training queued"
    )
    work = application.lab.get("test")
    assert cb["poll"](0, 0, "first_principles", plan, "test", 0)[0] is no_update
    for state in ("queued", "running", "done", "failed"):
        work.job = {"status": state, "completed": 1, "starts": 2, "error": "problem"}
        work.version = 2
        result = cb["poll"](0, 0, "first_principles", plan, "test", 0)
        assert result[0] == 2
        assert result[2] == (state in ("queued", "running"))
        assert result[4] == (state not in ("queued", "running"))
    work.job = {"status": "done", "converged": False}
    assert "budget" in cb["poll"](0, 0, "first_principles", plan, "test", 2)[1]
    assert not cb["poll"](0, 0, "hybrid", plan, "test", 2)[2]
    assert cb["poll"](0, 0, "first_principles", None, "test", 2)[2]


def test_recipe_loading_editing_and_scenario_callbacks(application, monkeypatch):
    cb = application.twin_callbacks
    trigger(monkeypatch, "reference-run")
    recipe = cb["recipe"](0, 0, 1, [], "test")
    assert recipe[1] == 4
    assert cb["recipe"](0, 0, 99, [], "test")[0] == []
    trigger(monkeypatch, "add-event")
    assert len(cb["recipe"](0, 1, 1, recipe[0], "test")[0]) == 2
    settings = scenario_defaults(application.lab.get("test").runs[1])
    args = [
        1,
        0,
        0,
        1,
        recipe[0],
        settings["duration"],
        settings["volume"],
        1,
        1,
        1,
        *[settings[n] for n in ("VCD", "TCD", "GLC", "GLU", "LAC")],
        "test",
    ]
    trigger(monkeypatch, "simulate")
    result, message = cb["simulation"](*args)
    assert "complete" in message
    np.testing.assert_allclose(result["values"], result["baseline"])
    assert not cb["simulation_results"](result, "GLC")[2]
    assert cb["simulation_results"](None, "VCD")[2]
    invalid = args.copy()
    invalid[6] = 0
    assert cb["simulation"](*invalid)[0] is no_update
    application.lab.get("test").parameters = None
    assert "Train" in cb["simulation"](*args)[1]
    trigger(monkeypatch, "model-version")
    assert cb["simulation"](*args)[0] is None


def test_recipe_simulation_with_mechanistic_nitrogen(application, monkeypatch):
    from dataclasses import replace

    work = application.lab.get("test")
    run = work.runs[1]
    work.runs[1] = replace(
        run,
        additional={
            name: np.full(len(run.times), value) for name, value in (("GLN", 0.02), ("AMM", 0.01))
        },
    )
    work.nitrogen = Nitrogen(0.4, 0.02, 0.05)
    cb = application.twin_callbacks
    trigger(monkeypatch, "reference-run")
    rows, duration, volume, strength, *initial = cb["recipe"](0, 0, 1, [], "test")
    gln, amm, age = cb["hybrid_initial"](1, 0, "test")
    assert (gln, amm) == (0.02, 0.01)
    trigger(monkeypatch, "simulate")
    result, message = cb["simulation"](
        1, 0, 0, 1, rows, duration, volume, strength, 1, 1, *initial, "test", gln, amm, age
    )
    assert "complete" in message
    assert result["columns"][-2:] == ["GLN", "AMM"]
    np.testing.assert_allclose(result["values"], result["baseline"])
    assert "ammonia_growth" in str(cb["training_results"](0, 0, "GLN", [1], "test")[1])


def test_exports_and_chart_variants(application, monkeypatch):
    cb = application.twin_callbacks["download"]
    trigger(monkeypatch, "download-model")
    assert cb(0, 1, 0, "test", None)["filename"] == "bioreactor_model.json"
    trigger(monkeypatch, "download-data")
    assert cb(1, 0, 0, "test", None)["filename"] == "observations.csv"
    trigger(monkeypatch, "download-simulation")
    result = {"values": [[1, 2, 3, 4, 5]], "volume": [199]}
    assert cb(0, 0, 1, "test", result)["filename"] == "simulation.csv"
    assert cb(0, 0, 1, "test", None)["filename"] == "observations.csv"
    work = application.lab.get("test")
    assert len(comparison(work, [1], "VCD").data) == 2
    work.predictions = {}
    assert len(comparison(work, [1], "VCD").data) == 1
    assert len(exploration(work.runs, [], "VCD").data) == 0
    info = replicate_analysis(work.runs, [1, 2])
    upper, lower, first, second, mean = replicate_figure(info, "GLC", [1, 2]).data
    assert upper.mode == lower.mode == "lines"
    assert upper.legendgroup == lower.legendgroup
    assert mean.name == "Observed pair mean"
    np.testing.assert_allclose(mean.y, (np.asarray(first.y) + np.asarray(second.y)) / 2)
    assert len(simulation_figure(np.ones((2, 5)), np.ones((2, 5)), "TCD").data) == 2


def test_app_entry_and_default_workspace(tmp_path, monkeypatch):
    import dash

    calls = []
    monkeypatch.setattr(dash.Dash, "run", lambda self, **kwargs: calls.append(kwargs))
    from bioreactor_twin.web.app import main

    main(["--port", "8099"])
    assert calls[-1]["host"] == "127.0.0.1"
    monkeypatch.setattr(sys, "argv", ["lab", "--port", "8098"])
    runpy.run_module("bioreactor_twin.web.app", run_name="__main__")
    assert calls[-1]["port"] == 8098
    app = create_app(root=tmp_path)
    assert app.lab.get("empty").runs == {}
    app.lab.executor.shutdown(wait=True)

"""Independent disturbance causality, controller constraints and adapter contracts."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from bioreactor_twin.control.dynamics import (
    Settings,
    advance,
    forecast,
    reachability,
    target_profile,
)
from bioreactor_twin.control.experiments import experiment, trial
from bioreactor_twin.control.mpc import choose_dose
from bioreactor_twin.control.observer import Observer
from bioreactor_twin.control.web import DEFAULTS, figure, review
from bioreactor_twin.models.kinetics import Parameters
from bioreactor_twin.simulation.engine import initialize, simulate
from bioreactor_twin.web.app import create_app
from bioreactor_twin.web.workspace import Lab, Workspace
from dash import no_update
from test_numerics import synthetic_run


def setup():
    r = replace(synthetic_run(), times=np.array([0]), observed=synthetic_run().observed[:1])
    s = Settings(horizon=2, interval=1, assay_interval=1, window=2, disturbance=1)
    return r, Parameters(), s


@pytest.mark.parametrize(
    "kw",
    [
        {"delay": -0.1},
        {"horizon": 0},
        {"horizon": 97},
        {"noise": float("nan")},
        {"multiplier": 0},
        {"max_glucose": 0},
    ],
)
def test_invalid_settings(kw):
    with pytest.raises(ValueError):
        replace(Settings(), **kw).validate()


def test_amount_forecast_and_reachability():
    r, p, s = setup()
    state = initialize(r)
    np.testing.assert_allclose(forecast(state, p, r.events[:-1], r.strength), simulate(r, p))
    assert advance(state, p, np.zeros(4), factor=0.5)[1] < advance(state, p, np.zeros(4))[1]
    assert reachability(state, p, 0.01)["feasible"]
    assert not reachability(state, p, 1)["feasible"]
    state[1] = p.capacity * state[0] * 2
    assert not reachability(state, p, 0)["feasible"]
    rows = [dict(hour=0, VCD=0.3), dict(hour=4, VCD=0.5)]
    assert target_profile(rows, 4, 0.3)[-1] == 0.5
    with pytest.raises(ValueError):
        target_profile([], 4, 0.3)
    with pytest.raises(ValueError):
        target_profile(rows, 4, 0.4)
    dose, info = choose_dose(initialize(r), p, r.events[:2], 1, np.array([0.31, 0.32]), s)
    assert 0 <= dose <= s.max_dose_ml and "success" in info
    assert choose_dose(initialize(r), p, [], 1, [], s)[0] == 0
    dose, info = choose_dose(
        initialize(r), p, r.events[:2], 1, np.ones(2), replace(s, max_volume_ml=1)
    )
    assert dose == 0 and not info["success"]


def test_causal_assays_and_forecasts():
    r, p, s = setup()
    a = experiment(r, p, s)
    b = experiment(r, p, replace(s, disturbance=3))
    one = a["cases"]["Growth disturbance"]
    two = b["cases"]["Growth disturbance"]
    np.testing.assert_allclose(one["truth"][:2], two["truth"][:2])
    assert one["updates"]["Adaptive"][0]["hour"] == 1
    delayed = trial(r, p, replace(s, delay=1, noise=0.1), adaptive=True)
    assert delayed["updates"]["Adaptive"][0]["hour"] == 2
    assert len(one["forecasts"]["Oracle"]) > 0
    target = np.array(simulate(r, p))[:, 0]
    c = experiment(r, p, s, target, "control")
    assert len(c["cases"]) == 4 and all(not v["scores"] for v in c["cases"].values())
    assert len(figure(c).data) > 0 and len(figure(a).data) > 0
    one.pop("maintenance_factors")
    assert len(figure(a).data) > 0
    assert review(c) is not None and review(a) is not None and review(None) is not None
    with pytest.raises(ValueError):
        experiment(r, p, s, mode="bad")
    with pytest.raises(ValueError):
        experiment(r, p, s, mode="control")
    with pytest.raises(ValueError):
        trial(replace(r, times=np.array([1])), p, s)
    observer = Observer(initialize(r), p, s)
    observer.history[0] = (observer.state.copy(), 1)
    assert observer.update(0, {0: np.ones(5)}, r.events, 1) is None


def test_web_adapters(tmp_path, monkeypatch):
    import dash

    r, p, s = setup()
    lab = Lab(tmp_path)
    lab.sessions["t"] = Workspace({7: r}, "x", p)
    cb = create_app(lab=lab).twin_callbacks
    assert cb["task2_visibility"]("recipe")[0]["display"] == "none"
    assert cb["task2_visibility"]("adaptation") == ({}, {})
    assert cb["task2_export"](1, None) is no_update
    args = [
        1,
        0,
        0,
        "adaptation",
        "t",
        *[getattr(s, k) for k in DEFAULTS],
        [],
        [],
        4,
        200,
        1,
        1,
        1,
        0.3,
        0.35,
        6,
        0.5,
        0.2,
    ]
    monkeypatch.setattr(dash, "ctx", SimpleNamespace(triggered_id="model-version"))
    assert cb["task2_run"](*args)[0] is None
    monkeypatch.setattr(dash, "ctx", SimpleNamespace(triggered_id="task2-run"))
    result, message = cb["task2_run"](*args)
    assert "complete" in message and cb["task2_export"](1, result)["content"]
    assert cb["task2_review"](result) is not None
    control_args = args.copy()
    control_args[3] = "control"
    control_args[17] = [dict(hour=0, VCD=0.3), dict(hour=4, VCD=0.4)]
    control_result, _ = cb["task2_run"](*control_args)
    assert control_result["mode"] == "control"
    lab.get("t").hybrid = True
    assert cb["task2_run"](*args)[0] is no_update
    lab.executor.shutdown(wait=True)


def test_numerical_rejection_paths_and_sparse_decisions(monkeypatch):
    import bioreactor_twin.control.observer as module

    r, p, s = setup()
    events = r.events[:2].copy()
    events[0, 0] = 1
    dose, status = choose_dose(initialize(r), p, events, 1, np.ones(2), s)
    assert dose == 0 and not status["success"]
    result = trial(r, p, replace(s, interval=2, max_volume_ml=1), np.full(5, 0.3), controlled=True)
    assert result["violations"]
    short = replace(r, events=r.events[:1], volumes=r.volumes[:1])
    assert len(trial(short, p, s)["truth"]) == 1
    real_advance = module.advance
    calls = [0]

    def failing_once(*a, **kw):
        calls[0] += 1
        if calls[0] == 1:
            raise ValueError("Invalid optimizer trial")
        return real_advance(*a, **kw)

    def optimizer(fun, x, **kw):
        assert np.all(fun(x) == 1e5)
        return SimpleNamespace(x=x, jac=np.zeros((10, len(x))), success=True, cost=0)

    monkeypatch.setattr(module, "advance", failing_once)
    monkeypatch.setattr(module, "least_squares", optimizer)
    observer = Observer(initialize(r), p, s)
    observer.history[0] = (observer.state.copy(), 1)
    observer.update(1, {0: np.ones(5), 1: np.ones(5)}, r.events, 1)
    assert observer.diagnostics[0]["converged"]


def test_forecast_replays_the_saved_plan_and_supports_long_assay_leads():
    r, p, settings = setup()
    result = trial(r, p, settings, np.full(len(r.events), 0.3), controlled=True)
    issued = result["forecasts"]["State correction"][0]
    assert issued["events"] == result["controller"][0]["planned_events"]
    np.testing.assert_allclose(
        forecast(
            np.array(issued["initial_state"]),
            Parameters(**issued["parameters"]),
            np.array(issued["events"]),
            r.strength,
            issued["growth_factor"],
        ),
        issued["values"],
    )
    long = replace(r, events=np.zeros((75, 4)), volumes=np.full(75, 0.2))
    result = trial(long, p, replace(settings, assay_interval=72, window=72, disturbance=0, delay=6))
    assert next(s for s in result["scores"] if s["lead_h"] == 72)["count"] == 1
    assert next(s for s in result["scores"] if s["lead_h"] is None)["count"] == 1


def test_bad_observer_update_retains_prior(monkeypatch):
    import bioreactor_twin.control.observer as module

    r, p, settings = setup()
    state = initialize(r)
    observer = Observer(state.copy(), p, settings)
    observer.history[0] = (state.copy(), 1.0)
    monkeypatch.setattr(
        module, "least_squares", lambda *a, **k: SimpleNamespace(success=False, cost=1)
    )
    observer.update(1, {1: r.observed[0]}, r.events, 1)
    np.testing.assert_array_equal(observer.state, state)
    assert not observer.diagnostics[-1]["accepted"]
    monkeypatch.setattr(
        module,
        "least_squares",
        lambda *a, **k: SimpleNamespace(success=True, cost=1, x=np.zeros(7)),
    )

    def invalid(*args, **kwargs):
        raise ValueError("Invalid trajectory")

    monkeypatch.setattr(module, "advance", invalid)
    observer.update(1, {1: r.observed[0]}, r.events, 1)
    assert not observer.diagnostics[-1]["accepted"]


def test_reference_target_and_reachability_boundaries():
    from bioreactor_twin.control.dynamics import reference_target
    from bioreactor_twin.models.kinetics import rates

    r, p, settings = setup()
    state = initialize(r)
    assert reachability(state, p, -rates(state, p)[1])["glucose_mM"] == 0
    state[1] = 2 * p.capacity * state[0]
    assert reachability(state, p, -rates(state, p)[1])["feasible"]
    assert len(reference_target(r, p, replace(settings, interval=2))) == len(r.events)
    assert len(reference_target(r, p, replace(settings, max_glucose=500))) == len(r.events)
    with pytest.raises(ValueError, match="Fixed feeds"):
        reference_target(r, p, replace(settings, max_volume_ml=1))
    with pytest.raises(ValueError, match="divide"):
        advance(state, p, r.events[0], step=0.3)

"""Independent conservation, recovery and whole-condition leakage tests."""

import json
from dataclasses import replace

import numpy as np
import pytest
from bioreactor_twin.hybrid.core import fit_core, trajectory
from bioreactor_twin.hybrid.operators import boundary, integrate
from bioreactor_twin.hybrid.regressors import RateModel, fit_rate, matern, scaler
from bioreactor_twin.hybrid.species import TARGETS, validate_targets
from bioreactor_twin.hybrid.training import configuration, fingerprint, scores, train_hybrid
from bioreactor_twin.hybrid.web import diagnostics, review
from bioreactor_twin.models.kinetics import Parameters
from bioreactor_twin.web.workspace import Lab, Workspace
from test_numerics import synthetic_run


def data(number=1):
    run = synthetic_run(number)
    run = replace(
        run, additional={"GLN": np.full(len(run.times), 0.03), "AMM": np.full(len(run.times), 0.01)}
    )
    trace = trajectory(run, Parameters())
    observed = run.observed.copy()
    extra = {}
    for name, q in zip(TARGETS, [0.001, 0.0002, 0.0001], strict=True):
        values = boundary(trace, name) + integrate(trace, np.full(len(trace.nodes), q))[:, 0]
        if name == "LAC":
            observed[:, 4] = values[run.times]
        else:
            extra[name] = values[run.times]
    return replace(run, observed=observed, additional=extra)


def test_integral_constant_rate_and_exact_sampling():
    run = data()
    trace = trajectory(run, Parameters())
    zero = integrate(trace, np.zeros(len(trace.nodes)))[:, 0]
    np.testing.assert_allclose(zero[run.times], 0)
    model = fit_rate({1: trace}, "GLN", penalty=1e-10)
    prediction, sd = model.predict(trace)
    np.testing.assert_allclose(prediction[run.times], run.additional["GLN"], atol=1e-9)
    np.testing.assert_allclose(model.rates(trace), 0.0002, atol=1e-9)
    assert sd is None
    # Uniform sampling preserves concentration of initial material, feeds dilute it.
    r = replace(run, events=np.zeros_like(run.events), volumes=np.full_like(run.volumes, 0.2))
    t = trajectory(r, Parameters())
    np.testing.assert_allclose(boundary(t, "AMM")[r.times], 0.01)
    np.testing.assert_allclose(boundary(t, "AMM", 0.02)[r.times], 0.02)
    fine = trajectory(run, Parameters(), 0.5)
    np.testing.assert_allclose(fine.values[run.times, :4], trace.values[run.times, :4], atol=1e-6)
    with pytest.raises(ValueError):
        trajectory(run, Parameters(), 0.3)


@pytest.mark.parametrize("family", ["ridge", "gp"])
def test_regressor_roundtrip_and_gp_covariance(family):
    run = data()
    t = trajectory(run, Parameters())
    m = fit_rate({1: t}, "AMM", ("mu",), family, penalty=0.001)
    exported = json.loads(json.dumps(m.export()))
    restored = RateModel(**exported)
    a, sd = restored.predict(t)
    np.testing.assert_allclose(a, m.predict(t)[0])
    assert (sd is not None) == (family == "gp")
    if sd is not None:
        assert np.min(sd[run.times]) >= 0
    assert m.rates(t).shape == (len(t.nodes),)
    m.coefficients = [-1e8] * len(m.coefficients)
    assert np.nanmin(m.predict(t)[0]) >= 0
    m.limit_depletion = False
    with pytest.raises(ValueError, match="negative"):
        m.predict(t)


def test_scalers_constant_and_correlated_features():
    nodes = np.ones((8, 6))
    nodes[:, 0] = np.arange(8)
    nodes[:, 1] = nodes[:, 0] * 2
    indices, _, _, removed = scaler(nodes, ("VCD", "viability", "mu"))
    assert indices == [0] and removed == ["viability", "mu"]
    np.testing.assert_allclose(np.diag(matern(nodes, nodes, 1)), 1)
    assert configuration("constant", ("mu",))[0]["family"] == "constant"
    assert len(configuration("auto", ("mu",))) == 20


def test_target_contracts():
    run = data()
    validate_targets({1: run}, TARGETS)
    for target in [[], ["other"]]:
        with pytest.raises(ValueError):
            validate_targets({1: run}, target)
    with pytest.raises(ValueError, match="no GLN"):
        validate_targets({1: synthetic_run()}, ["GLN"])
    with pytest.raises(ValueError, match="Invalid"):
        validate_targets({1: replace(run, additional={"GLN": np.array([-1])})}, ["GLN"])
    from bioreactor_twin.hybrid.species import overrides

    assert overrides([{"species": "GLN", "features": "mu, time", "regressor": "ridge"}])["GLN"][
        "features"
    ] == ("mu", "time")
    assert (
        overrides([{"species": "AMM", "features": "", "regressor": ""}])["AMM"]["family"]
        == "inherit"
    )
    for row in [
        {"species": "other"},
        {"species": "GLN", "features": "other"},
        {"species": "GLN", "regressor": "other"},
    ]:
        with pytest.raises(ValueError):
            overrides([row])


@pytest.fixture
def fast_core(monkeypatch):
    import bioreactor_twin.hybrid.training as module

    calls = []

    def fit(runs, starts, evaluations):
        calls.append(tuple(r.number for r in runs))
        return Parameters(), {"success": True, "cost": 0}

    monkeypatch.setattr(module, "fit_core", fit)
    return calls


def test_nested_cv_and_final_refit(fast_core):
    runs = {n: data(n) for n in (1, 2, 7, 8)}
    progress = []
    result = train_hybrid(
        runs, "cross_validation", options={"family": "constant"}, progress=progress.append
    )
    assert [f["test_runs"] for f in result["validation"]["folds"]] == [[1, 2], [7], [8]]
    assert result["trained"] == (1, 2, 7, 8)
    assert result["validation"]["prediction_source"] == "out_of_fold"
    assert all(
        not set(f["training_runs"]) & set(f["test_runs"]) for f in result["validation"]["folds"]
    )
    assert not result["invalid"][1]
    assert (1, 2) in fast_core and (1, 2, 7, 8) in fast_core
    exported = result["model"].export()
    assert exported["fingerprint"] == fingerprint(runs)
    work = Workspace(
        runs,
        "test",
        result["parameters"],
        result["trained"],
        predictions=result["predictions"],
        validation=result["validation"],
        hybrid=result["model"],
    )
    assert review(work, "rates", [1], "GLN") is not None
    assert review(work, "features", [1], "LAC") is not None
    assert review(work, "diagnostics", [1], "GLN") is not None
    work.hybrid = None
    assert "Train" in str(review(work, "features", [1], "LAC"))


@pytest.mark.parametrize("feature_mode", ["fixed", "manual", "auto"])
def test_feature_modes_and_train_test(fast_core, feature_mode):
    runs = {n: data(n) for n in (1, 2, 7)}
    result = train_hybrid(
        runs,
        test_runs=(7,),
        options={
            "targets": ["GLN"],
            "family": "ridge",
            "feature_mode": feature_mode,
            "features": ["mu"],
        },
    )
    assert result["trained"] == (1, 2)
    assert result["model"].species["GLN"].family in ("ridge", "constant")
    one = train_hybrid(
        {1: runs[1]}, options={"targets": ["GLN"], "family": "ridge", "feature_mode": "auto"}
    )
    assert one["model"].species["GLN"].family == "constant"
    specific = train_hybrid(
        runs,
        options={
            "targets": ["GLN"],
            "family": "auto",
            "feature_mode": "manual",
            "species_settings": [{"species": "GLN", "features": "mu", "regressor": "ridge"}],
        },
    )
    assert specific["model"].species["GLN"].family == "ridge"
    for options in [{"family": "bad"}, {"feature_mode": "bad"}, {"features": ["bad"]}]:
        with pytest.raises(ValueError):
            train_hybrid(runs, options=options)


def test_future_assays_do_not_enter_predictor(fast_core):
    run = data()
    result = train_hybrid({1: run}, options={"family": "constant"})
    before = result["model"].simulate(run)[0]
    changed = replace(
        run,
        observed=run.observed.copy(),
        additional={k: v.copy() for k, v in run.additional.items()},
    )
    changed.observed[1:] = 999
    for v in changed.additional.values():
        v[1:] = 999
    np.testing.assert_allclose(result["model"].simulate(changed)[0], before, equal_nan=True)
    model = result["model"]
    model.species["GLN"].coefficients = [-1e8]
    model.species["GLN"].limit_depletion = False
    predicted, _, invalid = model.simulate(run)
    assert "GLN" in invalid and scores({1: run}, {1: predicted}, ["GLN"])[0]["GLN"] is None


def test_real_core_fit_and_recovery():
    p, diag = fit_core([data()], 2, 2)
    assert np.isfinite(diag["cost"]) and p.mu_max > 0


def test_nonconstant_rate_feature_search(fast_core):
    runs = {}
    for n in (1, 2, 7, 8):
        r = data(n)
        t = trajectory(r, Parameters())
        concentration = boundary(t, "GLN") + integrate(t, 0.01 * (1 + t.nodes[:, 5] * 2))[:, 0]
        runs[n] = replace(r, additional={**r.additional, "GLN": concentration[r.times]})
    result = train_hybrid(
        runs, options={"targets": ["GLN"], "family": "ridge", "feature_mode": "auto"}
    )
    assert result["model"].selection["GLN"]["selected"]["features"]
    limited = train_hybrid(
        {n: runs[n] for n in (1, 2, 7)},
        options={"targets": ["GLN"], "family": "ridge", "feature_mode": "auto"},
    )
    assert len(limited["model"].selection["GLN"]["selected"]["features"]) == 1
    restored = type(result["model"]).restore(json.loads(json.dumps(result["model"].export())))
    np.testing.assert_allclose(
        restored.simulate(runs[1])[0], result["model"].simulate(runs[1])[0], equal_nan=True
    )


def test_no_valid_candidate_preserves_failure(fast_core, monkeypatch):
    import bioreactor_twin.hybrid.training as module

    monkeypatch.setattr(
        module, "fit_rate", lambda *a, **k: (_ for _ in ()).throw(ValueError("invalid rate"))
    )
    with pytest.raises(ValueError, match="No physically valid"):
        train_hybrid({1: data()}, options={"targets": ["GLN"]})


def test_core_invalid_trial_penalty_and_multistart(monkeypatch):
    import bioreactor_twin.hybrid.core as module

    real = module.trajectory
    calls = []

    def probe(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("invalid trial")
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "trajectory", probe)
    assert np.isfinite(module.fit_core([data()], 2, 2)[1]["cost"])
    zero = np.zeros(6)
    zero[0] = 0.2
    from bioreactor_twin.hybrid.core import features

    assert features(zero, Parameters(), 0)[1] == 0


def test_hybrid_app_workflow_saved_fit_and_scenario(tmp_path, fast_core, monkeypatch):
    from types import SimpleNamespace

    import bioreactor_twin.web.callbacks as callbacks
    from bioreactor_twin.web.app import create_app
    from bioreactor_twin.web.charts import comparison, exploration, simulation_figure
    from bioreactor_twin.web.services import (
        recipe_from_run,
        replicate_analysis,
        scenario_defaults,
        scenario_run,
    )

    runs = {n: data(n) for n in (1, 2, 7, 8)}
    result = train_hybrid(runs, "cross_validation", options={"family": "constant"})
    lab = Lab(tmp_path)
    work = Workspace(runs, "synthetic")
    lab.sessions["hybrid"] = work
    import bioreactor_twin.web.workspace as workspace

    monkeypatch.setattr(workspace, "train_hybrid", lambda *a, **k: result)
    work.job = {"status": "idle", "starts": 4}
    lab._fit(work, tuple(runs), 1, 10, "cross_validation", model_type="hybrid")
    assert work.hybrid is not None and work.job["status"] == "done"
    app = create_app(lab=lab)
    cb = app.twin_callbacks
    assert cb["hybrid_controls"]("hybrid") == {}
    assert cb["hybrid_controls"]("first_principles") == {"display": "none"}
    assert cb["hybrid_initial"](1, 0, "hybrid") == (0.03, 0.01, 0.0)
    assert cb["hybrid_initial"](99, 0, "hybrid") == (0.0, 0.0, 0.0)
    assert cb["hybrid_review"](0, 0, "rates", [1], "GLN", [], "hybrid") is not None
    assert "Could not" in cb["hybrid_load"](1, "hybrid")[0]
    folder = tmp_path / "outputs/hybrid_training"
    folder.mkdir(parents=True)
    (folder / "model.json").write_text(json.dumps(result["model"].export()))
    (folder / "validation.json").write_text(json.dumps(result["validation"]))
    np.savez(folder / "predictions.npz", **{str(n): v for n, v in result["predictions"].items()})
    assert "Loaded" in cb["hybrid_load"](1, "hybrid")[0]
    assert json.loads(lab.model_json("hybrid"))["hybrid"]["species"]
    work.job = {"status": "running"}
    with pytest.raises(ValueError, match="finish"):
        lab.load_hybrid("hybrid")
    work.job = {"status": "idle"}
    info = replicate_analysis(runs, [1, 2], Parameters())
    assert len(info["sd"]) == 7
    assert len(cb["replicates"](0, "GLN", 0, 1, 2, 0, "hybrid")[1]) == 7
    defaults = scenario_defaults(runs[1])
    recipe = recipe_from_run(runs[1])
    monkeypatch.setattr(callbacks, "ctx", SimpleNamespace(triggered_id="simulate"))
    args = [
        1,
        0,
        0,
        1,
        recipe,
        defaults["duration"],
        defaults["volume"],
        1,
        1,
        1,
        *[defaults[n] for n in ("VCD", "TCD", "GLC", "GLU", "LAC")],
        "hybrid",
        0.03,
        0.01,
    ]
    scenario, message = cb["simulation"](*args)
    assert "complete" in message and len(scenario["values"][0]) == 7
    assert simulation_figure(np.ones((2, 5)), np.ones((2, 5)), "GLN").layout.annotations
    assert len(exploration(runs, [1], "GLN").data) == 1
    work.bands = {1: {"GLN": np.ones(len(runs[1].events)) * 0.01}}
    assert len(comparison(work, [1], "GLN").data) == 4
    bad = {**defaults, "GLN": float("nan")}
    with pytest.raises(ValueError, match="Initial glutamine and ammonia"):
        scenario_run(bad, recipe)
    # A physically invalid learned trajectory is exposed, not clipped or exported as success.
    work.hybrid.species["GLN"].coefficients = [-1e8]
    assert "Consumption limited" in cb["simulation"](*args)[1]
    work.hybrid.species["GLN"].limit_depletion = False
    assert "could not" in cb["simulation"](*args)[1]
    old = work.runs
    work.runs = {1: runs[1]}
    with pytest.raises(ValueError, match="does not match"):
        lab.load_hybrid("hybrid")
    work.runs = old
    # Tampered prediction bundles cannot be published.
    np.savez(
        folder / "predictions.npz", **{str(n): v + 1 for n, v in result["predictions"].items()}
    )
    with pytest.raises(ValueError, match="do not match"):
        lab.load_hybrid("hybrid")
    all_fit = train_hybrid(runs, options={"family": "constant"})
    (folder / "model.json").write_text(json.dumps(all_fit["model"].export()))
    (folder / "validation.json").write_text(json.dumps(all_fit["validation"]))
    np.savez(folder / "predictions.npz", **{str(n): v for n, v in all_fit["predictions"].items()})
    assert "Loaded" in lab.load_hybrid("hybrid")
    legacy = all_fit["model"]
    legacy.species["GLN"].coefficients = [-1e8]
    legacy.species["GLN"].limit_depletion = False
    (folder / "model.json").write_text(json.dumps(legacy.export()))
    old_predictions = {str(n): legacy.simulate(run)[0] for n, run in runs.items()}
    np.savez(folder / "predictions.npz", **old_predictions)
    assert "Loaded" in lab.load_hybrid("hybrid")
    for values in old_predictions.values():
        values[:, 5] = np.nan
    np.savez(folder / "predictions.npz", **old_predictions)
    assert "Loaded" in lab.load_hybrid("hybrid")
    assert np.isfinite(work.predictions[1][0, 5])
    with pytest.raises(ValueError, match="Choose"):
        lab.train("hybrid", list(runs), 1, 10, model_type="unknown")
    lab.executor.shutdown(wait=True)


def test_parity_masks_constant_targets_and_invalid_predictions(fast_core):
    runs = {n: data(n) for n in (1, 2)}
    result = train_hybrid(runs, options={"family": "constant"})
    work = Workspace(
        runs,
        "test",
        result["parameters"],
        (1, 2),
        predictions=result["predictions"],
        hybrid=result["model"],
        validation=result["validation"],
    )
    assert diagnostics(work, [1], "GLN")[1] == []
    assert diagnostics(work, [1], "GLN", True)[1]
    work.predictions[1][:, 5] = np.nan
    assert diagnostics(work, [1], "GLN", True)[1][0]["status"] == "Invalid trajectory"
    work.predictions[1][:, 5] = 0.03
    work.runs[1].additional["GLN"][:] = 0.03
    assert diagnostics(work, [1], "GLN", True)[1][0]["R2"] is None
    assert review(work, "rates", [1], "GLN") is not None


def test_scenario_culture_age_preserves_rate_features():
    from dataclasses import replace

    from bioreactor_twin.hybrid.core import trajectory
    from bioreactor_twin.web.services import scenario_defaults, scenario_run

    original = data(1)
    shifted = replace(original, age_offset_h=3.5)
    trace = trajectory(shifted, Parameters())
    ordinary = trajectory(original, Parameters())
    np.testing.assert_allclose(trace.nodes[:, 5], ordinary.nodes[:, 5] + 3.5)
    np.testing.assert_allclose(trace.values, ordinary.values)
    settings = scenario_defaults(shifted)
    assert scenario_run(settings, []).age_offset_h == 3.5
    assert fingerprint({1: original}) != fingerprint({1: shifted})
    for age in (-1, float("nan")):
        with pytest.raises(ValueError, match="culture age"):
            scenario_run({**settings, "culture_age": age}, [])


def test_invalid_incumbent_can_be_replaced(fast_core, monkeypatch):
    import bioreactor_twin.hybrid.training as module

    original = module.fit_rate

    def candidate(*args, **kwargs):
        if kwargs["family"] == "constant":
            raise ValueError("Invalid constant model")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "fit_rate", candidate)
    result = train_hybrid(
        {7: data(7), 8: data(8)},
        options=dict(targets=["GLN"], family="ridge", feature_mode="auto"),
    )
    assert result["model"].species["GLN"].family == "ridge"


def test_legacy_invalid_prefix_preserves_assay_score():
    run = data()
    trace = trajectory(run, Parameters())
    fit = fit_rate({1: trace}, "GLN")
    run = replace(
        run,
        times=run.times[:2],
        observed=run.observed[:2],
        additional={k: v[:2] for k, v in run.additional.items()},
    )
    trace = trajectory(run, Parameters())
    design = integrate(trace, fit.basis(trace.nodes))[:, 0]
    fit.coefficients = [-run.additional["GLN"][0] / ((design[1] + design[-1]) / 2)]
    fit.limit_depletion = False
    from bioreactor_twin.hybrid.training import HybridModel

    model = HybridModel(Parameters(), {"GLN": fit}, {}, "test")
    values, _, invalid = model.simulate(run)
    assert np.isnan(values[-1, 5]) and "hour" in invalid["GLN"]
    assert np.isfinite(scores({1: run}, {1: values}, ["GLN"])[0]["GLN"])


def test_training_positivity_constraint(monkeypatch):
    from types import SimpleNamespace

    import bioreactor_twin.hybrid.regressors as module

    run = data()
    run = replace(
        run,
        times=run.times[:2],
        observed=run.observed[:2],
        additional={**run.additional, "GLN": np.array([0.03, 0.0])},
    )
    trace = trajectory(run, Parameters())
    model = fit_rate({run.number: trace}, "GLN", family="gp")
    predicted, sd = model.predict(trace)
    assert np.nanmin(predicted) >= 0
    assert sd is None and model.covariance == []
    monkeypatch.setattr(module, "minimize", lambda *a, **kw: SimpleNamespace(success=False))
    with pytest.raises(ValueError, match="nonnegative rate fit failed"):
        fit_rate({run.number: trace}, "GLN")

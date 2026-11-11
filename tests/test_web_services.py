"""Real scientific service tests for uploads, repeatability and new feed recipes."""

import base64
from dataclasses import replace

import numpy as np
import pytest
from bioreactor_twin.calibration.fit import fit, weighted_loss
from bioreactor_twin.models.kinetics import Parameters
from bioreactor_twin.simulation.engine import simulate
from bioreactor_twin.web.services import (
    RECIPE,
    decode_upload,
    evaluate,
    observations,
    parameter_rows,
    recipe_events,
    recipe_from_run,
    replicate_analysis,
    run_summary,
    scenario_defaults,
    scenario_run,
)
from test_io_workflow import workbook
from test_numerics import synthetic_run


def test_upload_roundtrip_and_native_tables(tmp_path):
    path = workbook(tmp_path)
    content = "data:application/octet-stream;base64," + base64.b64encode(path.read_bytes()).decode()
    runs, digest = decode_upload(content, "study.xlsx")
    assert len(digest) == 64
    assert observations(runs).shape == (2, 10)
    assert observations(runs, []).empty
    assert run_summary(runs)[0]["B_strength"] == 1
    with pytest.raises(ValueError, match="xlsx"):
        decode_upload(content, "study.csv")
    with pytest.raises(ValueError, match="10 MB"):
        decode_upload("x," + "x" * 14_000_001, "study.xlsx")


def test_replicate_services_use_common_times_and_frozen_model():
    a = synthetic_run(1)
    b = replace(synthetic_run(2), observed=synthetic_run(2).observed * 1.01)
    runs = {1: a, 2: b}
    assert not replicate_analysis(runs, [1, 2])["adjusted"]
    assert replicate_analysis(runs, [1, 2], Parameters())["adjusted"]
    for pair in ([1], [1, 1]):
        with pytest.raises(ValueError, match="different"):
            replicate_analysis(runs, pair)
    with pytest.raises(ValueError, match="common"):
        replicate_analysis({1: a, 2: replace(b, times=np.array([0]))}, [1, 2])


def test_scenario_replays_reference_and_does_not_use_future_observations():
    source = synthetic_run()
    rows = recipe_from_run(source)
    defaults = scenario_defaults(source)
    scenario = scenario_run(defaults, rows)
    np.testing.assert_allclose(simulate(scenario, Parameters()), simulate(source, Parameters()))
    changed = scenario_run({**defaults, "scale_A": 2}, rows)
    assert changed.events[:, 1].sum() == 2 * scenario.events[:, 1].sum()
    assert changed.events[:, 0].sum() == scenario.events[:, 0].sum()
    strong = scenario_run({**defaults, "strength": 0.5}, rows)
    np.testing.assert_array_equal(strong.events, scenario.events)
    assert strong.strength == 0.5
    predictions, scores = evaluate({7: source}, Parameters())
    assert scores[0]["VCD"] == 0
    assert len(parameter_rows(Parameters())) == 14
    assert predictions[7].shape == (5, 5)


@pytest.mark.parametrize(
    "override",
    [
        {"duration": 0.5},
        {"duration": 501},
        {"GLC": None},
        {"volume": 0},
        {"VCD": 2, "TCD": 1},
        {"strength": 0},
        {"scale_A": -1},
        {"GLC": np.nan},
    ],
)
def test_invalid_operating_conditions_are_rejected(override):
    defaults = scenario_defaults(synthetic_run())
    with pytest.raises(ValueError):
        scenario_run({**defaults, **override}, [])


@pytest.mark.parametrize("hour,amount", [(0.5, 1), (5, 1), (0, -1), (0, np.nan)])
def test_invalid_recipe_entries_raise(hour, amount):
    row = dict.fromkeys(RECIPE, 0)
    row.update(hour=hour, feed_A_mL=amount)
    with pytest.raises(ValueError):
        recipe_events(4, [row])


def test_cumulative_sampling_and_simultaneous_events():
    row = dict.fromkeys(RECIPE, 0)
    row.update(hour=1, sample_mL=200)
    with pytest.raises(ValueError, match="empty"):
        scenario_run(scenario_defaults(synthetic_run()), [row])
    row.update(sample_mL=1)
    assert recipe_events(4, [row, row])[1, 0] == 0.002


def test_group_balanced_loss_keeps_replicate_group_weight_and_threshold():
    z = np.array([0.0, 1.0, 9.0])
    loss = weighted_loss(np.array([0.5, 0.5, 1.0]))(z)
    np.testing.assert_allclose(loss[0], np.array([0.5, 0.5, 1.0]) * 2 * (np.sqrt(1 + z) - 1))
    a = synthetic_run(1)
    records = []
    _, info = fit(
        [a, replace(a, number=2), replace(a, number=7)],
        starts=1,
        max_nfev=2,
        grouped=True,
        on_progress=records.append,
    )
    assert info["cost"] == 0
    assert len(records) == 1

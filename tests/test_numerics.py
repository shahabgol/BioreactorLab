"""Independent conservation, analytical-solution and event-side checks."""

from dataclasses import replace

import numpy as np
import pytest
from bioreactor_twin.data.dataset import FEED_A, FEED_B, Run
from bioreactor_twin.models.kinetics import Parameters, derivative, monod, rates
from bioreactor_twin.simulation.engine import (
    event,
    initialize,
    observe,
    rk4,
    simulate,
    validate_state,
)


def synthetic_run(number=7):
    events = np.zeros((5, 4))
    events[2] = [0.001, 0.001, 0, 0]
    observed = np.tile([0.3, 0.35, 6, 0.5, 0.2], (5, 1))
    run = Run(number, events, np.full(5, 0.2), np.arange(5), observed, tuple(range(2, 7)))
    return replace(run, observed=simulate(run, Parameters()))


@pytest.mark.parametrize("kwargs", [{"mu_max": 0}, {"lysis": -1}, {"k_g": np.nan}])
def test_parameter_validation(kwargs):
    with pytest.raises(ValueError):
        Parameters(**kwargs)


def test_sample_preserves_all_concentrations_and_feed_conserves_amounts():
    state = np.array([0.2, 0.06, 0.01, 5, 1, 0.5])
    sampled = event(state, [0.002, 0, 0, 0])
    np.testing.assert_allclose(observe(sampled), observe(state), atol=1e-12)
    fed = event(state, [0, 0.001, 0.008, 0.003], strength=0.5)
    assert fed[0] == pytest.approx(0.212)
    np.testing.assert_allclose(fed[1:3], state[1:3])
    np.testing.assert_allclose(fed[3:] - state[3:], 0.001 * FEED_A + 0.008 * 0.5 * FEED_B)
    np.testing.assert_array_equal(state, [0.2, 0.06, 0.01, 5, 1, 0.5])


@pytest.mark.parametrize("volumes", [[0.2, 0, 0, 0], [0, -0.1, 0, 0]])
def test_impossible_events_raise(volumes):
    with pytest.raises(ValueError):
        event([0.2, 1, 1, 1, 1, 1], volumes)


@pytest.mark.parametrize(
    "state", [[0, 1, 1, 1, 1, 1], [1, -1, 1, 1, 1, 1], [1, 1, 1, np.nan, 1, 1]]
)
def test_nonphysical_states_raise(state):
    with pytest.raises(ValueError):
        validate_state(state)


def test_zero_substrate_boundaries_and_death_transfer():
    assert monod(-1, 1) == 0
    assert monod(1, 1) == 0.5
    state = [1, 2, 3, 0, 0, 0]
    p = Parameters(lysis=0.01)
    growth, death, g, e, lactate = rates(state, p)
    assert (growth, g, e, lactate) == (0, 0, 0, 0)
    changes = derivative(state, p)
    assert changes[1] + changes[2] == pytest.approx(-0.01 * 3)
    assert death > 0


def test_exponential_growth_and_dead_cell_closed_form():
    p = Parameters(
        mu_max=0.03,
        k_g=1e-12,
        k_e=1e-12,
        capacity=1e20,
        death=0.004,
        death_density=0,
        lysis=0.002,
        glucose_growth=0,
        glucose_maintenance=0,
        glutamate_growth=0,
        glutamate_uptake=0,
        lactate_yield=0,
        lactate_uptake=0,
    )
    state = [1, 1, 0.2, 100, 100, 1]
    for _ in range(100):
        state = rk4(state, p, 0.1)
    expected_v = np.exp(0.026 * 10)
    expected_d = 0.2 * np.exp(-0.002 * 10) + 0.004 * (expected_v - np.exp(-0.002 * 10)) / 0.028
    np.testing.assert_allclose(state[1:3], [expected_v, expected_d], rtol=1e-10)


def test_observation_before_feed_and_no_later_assay_reset():
    run = synthetic_run()
    predicted = simulate(run, Parameters())
    altered = run.observed.copy()
    altered[1:] *= 2
    np.testing.assert_allclose(simulate(replace(run, observed=altered), Parameters()), predicted)
    np.testing.assert_allclose(observe(initialize(run)), run.observed[0])
    assert predicted[3, 2] > predicted[2, 2]  # the hour-2 feed appears after its assay


@pytest.mark.parametrize("step", [0, -0.1, 0.3])
def test_invalid_steps_raise(step):
    with pytest.raises(ValueError):
        simulate(synthetic_run(), Parameters(), step=step)


def test_hourly_rk4_matches_refinement_and_independent_solver():
    run = synthetic_run()
    reference = simulate(run, Parameters(), reference=True)
    for step in [1, 0.5, 0.25]:
        np.testing.assert_allclose(simulate(run, Parameters(), step), reference, rtol=1e-7)


def test_reference_solver_failure_is_reported(monkeypatch):
    from types import SimpleNamespace

    import bioreactor_twin.simulation.engine as engine

    monkeypatch.setattr(
        engine, "solve_ivp", lambda *a, **k: SimpleNamespace(success=False, message="solver failed")
    )
    with pytest.raises(ValueError, match="solver failed"):
        simulate(synthetic_run(), Parameters(), reference=True)


def test_depletion_retries_stay_inside_the_hour():
    run = synthetic_run()
    run.events[:] = 0
    run.observed[0, :3] = [20, 20, 0.01]
    parameters = replace(Parameters(), glucose_maintenance=0.8, k_g=2)
    reference = simulate(run, parameters, reference=True)
    prediction = simulate(run, parameters)
    assert np.all(prediction[:, 2] >= 0)
    np.testing.assert_allclose(prediction[:, 2], reference[:, 2], atol=3e-5)
    from bioreactor_twin.hybrid.core import trajectory

    trace = trajectory(run, parameters)
    np.testing.assert_allclose(trace.values[:, :4], prediction[:, :4])
    assert len(trace.exposure) > 4 * (len(run.events) - 1)


def test_step_retry_limit_is_reported():
    from bioreactor_twin.simulation.engine import InvalidState, refine_step

    def invalid(*args):
        raise InvalidState("failed stage")

    with pytest.raises(InvalidState, match="failed stage"):
        refine_step(initialize(synthetic_run()), 0, 1, invalid, retries=0)

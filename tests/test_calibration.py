"""Synthetic calibration and information-barrier checks; no private workbook needed."""

from dataclasses import replace

import numpy as np
import pytest
from bioreactor_twin.calibration.fit import KEYS, fit, residuals, unpack
from bioreactor_twin.models.kinetics import Parameters
from bioreactor_twin.validation.uncertainty import envelope, metrics, repeatability
from test_numerics import synthetic_run


def test_synthetic_parameter_fit_and_diagnostics():
    run = synthetic_run()
    p, info = fit([run], starts=2, max_nfev=2)
    assert info["cost"] < 1e-12
    assert p.mu_max == pytest.approx(Parameters().mu_max)
    assert len(info["attempts"]) == 2
    assert len(info["jacobian_singular_values"]) == len(KEYS)


def test_fit_rejects_unbalanced_data_and_no_starts():
    run = synthetic_run()
    with pytest.raises(ValueError, match="equal"):
        fit([run, replace(run, times=run.times[:-1])])
    with pytest.raises(ValueError, match="one calibration"):
        fit([run], starts=0)


def test_bad_trajectory_retains_residual_dimension():
    p = Parameters()
    vector = np.log([getattr(p, key) for key in KEYS])
    run = synthetic_run()
    run.events[2, 0] = 1
    result = residuals(vector, [run], {})
    assert result.shape == (20,)
    assert np.all(result == 1e4)
    assert unpack(np.log([getattr(p, key) for key in KEYS])).lysis == 0


def test_paired_repeatability_has_known_independent_difference_scale():
    a = synthetic_run(1)
    b = replace(a, number=2, observed=a.observed.copy())
    pa = a.observed.copy()
    pb = a.observed.copy()
    altered = a.observed.copy()
    altered[1:] *= 1.2
    info = repeatability(replace(a, observed=altered), b, {1: pa, 2: pb})
    expected = 0.2 * np.sqrt(np.mean(pa[1:] ** 2, axis=0) / 2)
    np.testing.assert_allclose(info["sd"], expected)
    assert metrics(b, pb)["rmse"] == [0] * 5
    low, high = envelope(pa, np.array(info["sd"]))
    assert np.all(low >= 0)
    assert np.all(low < pa)
    assert np.all(high > pa)

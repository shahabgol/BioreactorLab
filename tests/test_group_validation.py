"""Whole-condition leakage checks, real grouped CV, and split-selection semantics."""

from dataclasses import replace

import numpy as np
import pytest
from bioreactor_twin.calibration.fit import fit
from bioreactor_twin.calibration.validation import (
    condition_groups,
    linked_test_rows,
    train_validated,
    validation_splits,
)
from bioreactor_twin.models.kinetics import Parameters
from test_numerics import synthetic_run


def test_leave_one_condition_out_keeps_replicates_together():
    numbers = [1, 2, 7, 8]
    assert condition_groups(numbers) == [(1, 2), (7,), (8,)]
    splits = validation_splits(numbers, "cross_validation")
    assert [s.test for s in splits] == [(1, 2), (7,), (8,)]
    for split in splits:
        assert not set(split.training) & set(split.test)
        assert set(split.training) | set(split.test) == set(numbers)


@pytest.mark.parametrize(
    "numbers,mode,test",
    [
        ([], "train_test", []),
        ([1, 2], "cross_validation", []),
        ([1, 2, 7], "bad", []),
        ([1, 2, 7], "train_test", [99]),
        ([1, 2, 7], "train_test", [1]),
        ([1, 2, 7], "train_test", [1, 2, 7]),
    ],
)
def test_invalid_validation_designs_raise(numbers, mode, test):
    with pytest.raises(ValueError):
        validation_splits(numbers, mode, test)


def test_linked_checkboxes_add_and_remove_the_pair():
    numbers = [1, 2, 7, 8]
    assert linked_test_rows(numbers, [0], []) == [0, 1]
    assert linked_test_rows(numbers, [1], [0, 1]) == []
    assert linked_test_rows(numbers, [0, 1, 2], [0, 1]) == [0, 1, 2]
    assert linked_test_rows(numbers, [0, 1], [0, 1]) == [0, 1]


def test_cv_predictions_use_fold_models_not_the_final_refit():
    runs = {n: synthetic_run(n) for n in (1, 2, 7, 8)}
    calls, progress = [], []

    def fitter(training, **kwargs):
        ids = tuple(r.number for r in training)
        calls.append(ids)
        kwargs["on_progress"]({"start": 0, "cost": 0})
        return Parameters(mu_max=0.03 + 0.001 * sum(ids)), {"success": True}

    result = train_validated(runs, "cross_validation", (), 1, 2, fitter, progress.append)
    assert calls == [(7, 8), (1, 2, 8), (1, 2, 7), (1, 2, 7, 8)]
    assert [p["start"] for p in progress] == [0, 1, 2, 3]
    assert result["parameters"].mu_max == pytest.approx(0.048)
    fold = result["validation"]["folds"][0]
    assert fold["parameters"]["mu_max"] == pytest.approx(0.045)
    from bioreactor_twin.simulation.engine import simulate

    np.testing.assert_allclose(
        result["predictions"][1], simulate(runs[1], Parameters(mu_max=0.045))
    )
    assert not np.allclose(result["predictions"][1], simulate(runs[1], result["parameters"]))
    assert result["validation"]["prediction_source"] == "out_of_fold"
    assert result["trained"] == (1, 2, 7, 8)


def test_held_out_values_do_not_enter_the_training_fold():
    runs = {n: synthetic_run(n) for n in (1, 2, 7, 8)}
    calls = []

    def fitter(training, **kwargs):
        calls.append([r.number for r in training])
        assert all(np.max(r.observed) < 100 for r in training)
        return Parameters(), {"success": True}

    for number in (1, 2):
        observed = runs[number].observed.copy()
        observed[1:] = 999
        runs[number] = replace(runs[number], observed=observed)
    result = train_validated(runs, "train_test", [1, 2], 1, 2, fitter, lambda _: None)
    assert calls == [[7, 8]]
    assert result["trained"] == (7, 8)
    assert len(result["validation"]["folds"][0]["scores"]) == 2


def test_real_synthetic_cross_validation_and_all_data_refit():
    runs = {n: synthetic_run(n) for n in (1, 2, 7, 8)}
    result = train_validated(runs, "cross_validation", (), 1, 2, fit, lambda _: None)
    assert result["converged"]
    assert all(row["VCD"] < 1e-9 for row in result["scores"])
    refit = train_validated(runs, "train_test", (), 1, 2, fit, lambda _: None)
    assert refit["validation"]["folds"][0]["test_runs"] == []

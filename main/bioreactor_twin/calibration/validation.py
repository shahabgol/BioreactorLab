"""Whole-experiment validation with replicate groups isolated from fitting."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from bioreactor_twin.calibration.nitrogen import fit_nitrogen, nitrogen_scores
from bioreactor_twin.data.dataset import NAMES, condition_key
from bioreactor_twin.simulation.nitrogen import simulate_mechanistic
from bioreactor_twin.validation.uncertainty import metrics


@dataclass(frozen=True)
class Split:
    training: tuple[int, ...]
    test: tuple[int, ...]


def condition_groups(numbers):
    groups = {}
    for number in sorted(numbers):
        key = (
            condition_key(numbers[number])
            if isinstance(numbers, Mapping)
            else ("nominal" if number in (1, 2) else str(number))
        )
        groups.setdefault(key, []).append(number)
    return [tuple(group) for group in groups.values()]


def validation_splits(numbers, mode, test_runs=()):
    available = set(numbers)
    groups = condition_groups(numbers)
    if not available:
        raise ValueError("Load at least one experiment before training.")
    if mode == "cross_validation":
        if len(groups) < 2:
            raise ValueError("Cross-validation needs at least two independent conditions.")
        return [Split(tuple(sorted(available - set(group))), group) for group in groups]
    if mode != "train_test":
        raise ValueError("Choose train/test or grouped cross-validation.")
    test = set(test_runs)
    if not test <= available:
        raise ValueError("Test experiments must belong to the loaded dataset.")
    if any(test.intersection(group) and not set(group) <= test for group in groups):
        raise ValueError(
            "Replicates must stay together: select every run in a replicate group, or none."
        )
    training = available - test
    if not training:
        raise ValueError("Keep at least one condition for training.")
    return [Split(tuple(sorted(training)), tuple(sorted(test)))]


def linked_test_rows(numbers, selected, previous, runs=None):
    """One checkbox action adds/removes an entire replicate condition."""
    selected = set(selected or [])
    previous = set(previous or [])
    added, removed = selected - previous, previous - selected
    for group in condition_groups(runs if runs is not None else numbers):
        indices = {numbers.index(number) for number in group}
        if indices & added:
            selected |= indices
        elif indices & removed:
            selected -= indices
    return sorted(selected)


def score_predictions(runs, predictions):
    return [
        {
            "run": n,
            **dict(
                zip(
                    NAMES,
                    metrics(runs[n], values[:, : len(NAMES)])["rmse"],
                    strict=True,
                )
            ),
            **nitrogen_scores(runs[n], values),
        }
        for n, values in predictions.items()
    ]


def train_validated(runs, mode, test_runs, starts, evaluations, fitter, progress):
    """Fit each validation fold independently, then refit all data after CV scoring.

    Out-of-fold curves come exclusively from models that did not fit that group.
    The final all-data model is for simulation, never for CV performance scoring.
    """
    splits = validation_splits(runs, mode, test_runs)
    predictions, folds, converged = {}, [], []

    def fit_stage(numbers, stage, offset):
        def report(record):
            progress({**record, "start": offset + record["start"], "stage": stage})

        core, diagnostics = fitter(
            [runs[n] for n in numbers],
            starts=starts,
            max_nfev=evaluations,
            grouped=True,
            on_progress=report,
        )
        nitrogen, nitrogen_fit = fit_nitrogen([runs[n] for n in numbers], core, evaluations)
        if nitrogen_fit is not None:
            diagnostics = {
                **diagnostics,
                "success": diagnostics["success"] and nitrogen_fit["success"],
                "nitrogen": nitrogen_fit,
            }
        return core, nitrogen, diagnostics

    for index, split in enumerate(splits):
        parameters, nitrogen, diagnostics = fit_stage(
            split.training, f"Fold {index + 1}/{len(splits)}", index * starts
        )
        converged.append(bool(diagnostics["success"]))
        targets = split.test if mode == "cross_validation" else tuple(runs)
        current = {n: simulate_mechanistic(runs[n], parameters, nitrogen) for n in targets}
        predictions.update(current)
        folds.append(
            {
                "fold": index + 1,
                "training_runs": list(split.training),
                "test_runs": list(split.test),
                "parameters": asdict(parameters),
                "nitrogen": asdict(nitrogen) if nitrogen is not None else None,
                "converged": bool(diagnostics["success"]),
                "scores": score_predictions(runs, {n: current[n] for n in split.test}),
                "diagnostics": diagnostics,
            }
        )
    trained = splits[0].training
    if mode == "cross_validation":
        trained = tuple(sorted(runs))
        parameters, nitrogen, diagnostics = fit_stage(
            trained, "Final all-data refit", len(splits) * starts
        )
        converged.append(bool(diagnostics["success"]))
    validation = {
        "mode": mode,
        "folds": folds,
        "final_training_runs": list(trained),
        "prediction_source": "out_of_fold" if mode == "cross_validation" else "split",
        "final_diagnostics": diagnostics,
    }
    return {
        "parameters": parameters,
        "nitrogen": nitrogen,
        "predictions": predictions,
        "scores": score_predictions(runs, predictions),
        "trained": trained,
        "converged": all(converged),
        "validation": validation,
    }

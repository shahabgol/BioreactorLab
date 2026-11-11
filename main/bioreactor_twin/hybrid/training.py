"""Nested, whole-condition fitting; core and preprocessing never see test assays."""

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from bioreactor_twin.calibration.validation import condition_groups, validation_splits
from bioreactor_twin.hybrid.core import fit_core, trajectory
from bioreactor_twin.hybrid.regressors import InvalidRateTrajectory, RateModel, fit_rate
from bioreactor_twin.hybrid.species import (
    FEATURES,
    OUTPUTS,
    PRESETS,
    TARGETS,
    assays,
    overrides,
    validate_targets,
)


@dataclass
class HybridModel:
    core: object
    species: dict
    selection: dict
    fingerprint: str

    @classmethod
    def restore(cls, artifact):
        from bioreactor_twin.models.kinetics import Parameters

        return cls(
            Parameters(**artifact["core"]),
            {k: RateModel(**v) for k, v in artifact["species"].items()},
            artifact["selection"],
            artifact["fingerprint"],
        )

    def export(self):
        return dict(
            core=asdict(self.core),
            species={k: v.export() for k, v in self.species.items()},
            selection=self.selection,
            fingerprint=self.fingerprint,
            gp_method="16-landmark Nystrom Matern-5/2 integral GP; conditional working-noise bands",
        )

    def simulate(self, run, *, diagnostics=None):
        trace = trajectory(run, self.core)
        values = np.column_stack((trace.values[:, :4], np.full((len(run.events), 3), np.nan)))
        uncertainty = {}
        invalid = {}
        for name, model in self.species.items():
            try:
                concentration, sd = model.predict(trace, diagnostics=diagnostics)
                values[:, OUTPUTS.index(name)] = concentration
                uncertainty[name] = sd
            except InvalidRateTrajectory as error:
                values[:, OUTPUTS.index(name)] = error.concentration
                uncertainty[name] = error.sd
                invalid[name] = str(error)
        return values, uncertainty, invalid


def fingerprint(runs):
    records = [
        (
            n,
            r.times.tolist(),
            r.observed.tolist(),
            r.events.tolist(),
            r.strength,
            r.age_offset_h,
            {k: np.asarray(v).tolist() for k, v in r.additional.items()},
        )
        for n, r in runs.items()
    ]
    payload = (
        dict(records=records, groups={n: r.replicate_group for n, r in runs.items()})
        if any(r.replicate_group is not None for r in runs.values())
        else records
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def scores(runs, predictions, targets):
    rows = []
    for n, values in predictions.items():
        row = {"run": n}
        for name in OUTPUTS:
            if name not in OUTPUTS[:4] and name not in targets:
                continue
            actual = (
                runs[n].observed[:, OUTPUTS.index(name)]
                if name in OUTPUTS[:5]
                else assays(runs[n], name)
            )
            error = values[runs[n].times[1:], OUTPUTS.index(name)] - actual[1:]
            row[name] = float(np.sqrt(np.mean(error**2))) if np.all(np.isfinite(error)) else None
        rows.append(row)
    return rows


def configuration(family, feature_names):
    if not feature_names or family == "constant":
        return [dict(family="constant", features=(), penalty=0.001, length=1.0)]
    families = ("ridge", "gp") if family == "auto" else (family,)
    return [
        dict(family=f, features=tuple(feature_names), penalty=p, length=length)
        for f in families
        for p in (0.1, 1.0, 10.0, 100.0)
        for length in ((0.5, 1.0, 2.0, 5.0) if f == "gp" else (1.0,))
    ]


def train_hybrid(
    runs,
    mode="train_test",
    test_runs=(),
    starts=1,
    evaluations=100,
    options=None,
    progress=lambda record: None,
):
    options = options or {}
    targets = tuple(options.get("targets", TARGETS))
    validate_targets(runs, targets)
    family = options.get("family", "auto")
    feature_mode = options.get("feature_mode", "fixed")
    if family not in ("auto", "constant", "ridge", "gp") or feature_mode not in (
        "fixed",
        "manual",
        "auto",
    ):
        raise ValueError("Invalid hybrid regressor or feature mode.")
    manual = tuple(options.get("features", FEATURES))
    per_species = overrides(options.get("species_settings"))
    if not set(manual) <= set(FEATURES):
        raise ValueError("Unknown hybrid feature.")
    splits = validation_splits(runs, mode, test_runs)
    cache = {}
    counter = 0

    def core(ids):
        nonlocal counter
        key = tuple(sorted(ids))
        if key not in cache:
            progress(dict(start=counter, cost=0, stage="Core: runs " + str(key)))
            p, diag = fit_core([runs[n] for n in key], starts, evaluations)
            cache[key] = (p, diag)
            counter += starts
        return cache[key]

    def traces(ids, p):
        return {n: trajectory(runs[n], p) for n in ids}

    def fit_partition(ids):
        p, diag = core(ids)
        full = traces(ids, p)
        models = {}
        reports = {}
        partition = {n: runs[n] for n in ids}
        groups = condition_groups(partition)
        inner = validation_splits(partition, "cross_validation") if len(groups) > 1 else []
        partitions = [
            (traces(s.training, core(s.training)[0]), traces(s.test, core(s.training)[0]))
            for s in inner
        ]
        for species in targets:
            setting = per_species.get(species, {})
            species_family = setting.get("family", "inherit")
            if species_family == "inherit":
                species_family = family
            evaluated = []

            def assess(config, species=species, evaluated=evaluated):
                errors = []
                try:
                    for training, validation in partitions:
                        candidate = fit_rate(training, species, **config)
                        group = []
                        for n, t in validation.items():
                            predicted, _ = candidate.predict(t)
                            group.append(
                                np.mean(
                                    (predicted[runs[n].times[1:]] - assays(runs[n], species)[1:])
                                    ** 2
                                )
                            )
                        errors.append(float(np.sqrt(np.mean(group))))
                    fitted = fit_rate(full, species, **config)
                    if not partitions:
                        errors = [
                            float(
                                np.sqrt(
                                    np.mean(
                                        (
                                            fitted.predict(t)[0][runs[n].times[1:]]
                                            - assays(runs[n], species)[1:]
                                        )
                                        ** 2
                                    )
                                )
                            )
                            for n, t in full.items()
                        ]
                    mean = float(np.mean(errors))
                except (ValueError, np.linalg.LinAlgError):
                    fitted = None
                    mean = float("inf")
                evaluated.append(
                    {**config, "rmse": mean if np.isfinite(mean) else None, "fold_errors": errors}
                )
                return mean, fitted, config, errors

            constant = assess(configuration("constant", ())[0])
            winner = constant
            if species_family != "constant":
                chosen = ()
                limit = 1 if len(groups) == 2 else 2
                if feature_mode == "auto" and len(groups) > 1:
                    for _ in range(limit):
                        candidates = [
                            assess(c)
                            for f in FEATURES
                            if f not in chosen
                            for c in configuration(species_family, (*chosen, f))
                        ]
                        best = min(candidates, key=lambda x: x[0])
                        if best[1] is not None and (
                            winner[1] is None
                            or (
                                best[0] < winner[0]
                                and best[0] <= winner[0] * 0.95
                                and all(
                                    a <= b * 1.1 for a, b in zip(best[3], winner[3], strict=True)
                                )
                            )
                        ):
                            winner = best
                            chosen = best[2]["features"]
                        else:
                            break
                elif feature_mode != "auto":
                    selected = (
                        PRESETS[species]
                        if feature_mode == "fixed"
                        else setting.get("features", manual)
                    )
                    candidates = [assess(c) for c in configuration(species_family, selected)]
                    winner = (
                        min([constant, *candidates], key=lambda x: x[0])
                        if species_family == "auto"
                        else min(candidates, key=lambda x: x[0])
                    )
            if winner[1] is None:
                raise ValueError(f"No physically valid {species} rate model for runs {ids}.")
            models[species] = winner[1]
            reports[species] = dict(
                selected=winner[2],
                candidates=evaluated,
                selection_rmse=winner[0],
                inner_groups=[list(s.test) for s in inner],
                note="Exploratory: inner fits may contain only one independent condition."
                if len(groups) < 3
                else "Training-only grouped selection.",
            )
            progress(
                dict(
                    start=counter,
                    cost=winner[0],
                    stage=f"{species}: {winner[2]['family']} selected",
                )
            )
        return HybridModel(p, models, reports, fingerprint({n: runs[n] for n in ids})), diag

    folds = []
    predictions = {}
    bands = {}
    invalid = {}
    depletion = {}
    converged = []
    for split in splits:
        model, diag = fit_partition(split.training)
        converged.append(diag["success"])
        current = {}
        for n in split.test if mode == "cross_validation" else runs:
            depletion[n] = {}
            current[n], bands[n], invalid[n] = model.simulate(runs[n], diagnostics=depletion[n])
        predictions.update(current)
        folds.append(
            dict(
                training_runs=list(split.training),
                test_runs=list(split.test),
                model=model.export(),
                diagnostics=diag,
                scores=scores(runs, current, targets),
            )
        )
    if mode == "cross_validation":
        model, diag = fit_partition(tuple(runs))
        converged.append(diag["success"])
    return dict(
        model=model,
        parameters=model.core,
        predictions=predictions,
        bands=bands,
        invalid=invalid,
        scores=scores(runs, predictions, targets),
        trained=tuple(runs) if mode == "cross_validation" else splits[0].training,
        converged=all(converged),
        validation=dict(
            mode=mode,
            folds=folds,
            prediction_source="out_of_fold" if mode == "cross_validation" else "split",
            depletion_hours={n: v for n, v in depletion.items() if v},
            invalid_forecasts={n: v for n, v in invalid.items() if v},
        ),
    )

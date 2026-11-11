"""Explicit hybrid target ownership and source-unit contracts."""

import numpy as np

TARGETS = ("LAC", "GLN", "AMM")
OUTPUTS = ("VCD", "TCD", "GLC", "GLU", *TARGETS)
FEATURES = ("VCD", "viability", "GLC", "GLU", "mu", "time")
PRESETS = {"LAC": ("GLC", "mu"), "GLN": ("mu",), "AMM": ("mu",)}
FLOORS = {"LAC": 0.05, "GLN": 0.005, "AMM": 0.002}


def overrides(rows):
    result = {}
    for row in rows or []:
        name = row["species"]
        features = tuple(x.strip() for x in (row.get("features") or "").split(",") if x.strip())
        family = row.get("regressor") or "inherit"
        if name not in TARGETS or not set(features) <= set(FEATURES):
            raise ValueError("Per-species settings contain an unknown species or feature.")
        if family not in ("inherit", "auto", "constant", "ridge", "gp"):
            raise ValueError("Per-species regressor must be inherit, auto, constant, ridge or gp.")
        result[name] = dict(features=features, family=family)
    return result


def assays(run, species):
    return run.observed[:, 4] if species == "LAC" else np.asarray(run.additional[species])


def validate_targets(runs, targets):
    if not targets or not set(targets) <= set(TARGETS):
        raise ValueError("Choose lactate, glutamine and/or ammonia.")
    for run in runs.values():
        for species in targets:
            try:
                values = assays(run, species)
            except KeyError as error:
                raise ValueError(f"Run {run.number} has no {species} assays.") from error
            if (
                values.shape != run.times.shape
                or not np.all(np.isfinite(values))
                or min(values) < 0
            ):
                raise ValueError(f"Invalid {species} assays for Run {run.number}.")

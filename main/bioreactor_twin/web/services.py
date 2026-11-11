"""UI-independent data exploration, replicate analysis and operating scenarios."""

import base64
import hashlib
from dataclasses import asdict
from io import BytesIO

import numpy as np
import pandas as pd

from bioreactor_twin.data.dataset import NAMES, Run, load_runs
from bioreactor_twin.simulation.engine import initialize, simulate
from bioreactor_twin.simulation.nitrogen import simulate_mechanistic
from bioreactor_twin.validation.uncertainty import metrics, repeatability

UNITS = {
    "VCD": "million cells/mL",
    "TCD": "million cells/mL",
    "GLC": "g/L",
    "GLU": "g/L",
    "LAC": "g/L",
    "GLN": "g/L",
    "AMM": "g/L",
}
LABELS = {
    "VCD": "Viable cell density",
    "TCD": "Total cell density",
    "GLC": "Glucose",
    "GLU": "Glutamate",
    "LAC": "Lactate",
    "GLN": "Glutamine",
    "AMM": "Ammonia (reported)",
}
RECIPE = ("hour", "sample_mL", "feed_A_mL", "feed_B_mL", "feed_C_mL")


def decode_upload(contents, filename):
    if not filename.lower().endswith(".xlsx"):
        raise ValueError("Choose an .xlsx workbook with the SampleData sheet.")
    encoded = contents.split(",", 1)[1]
    if len(encoded) > 14_000_000:
        raise ValueError("The workbook must be smaller than 10 MB.")
    payload = base64.b64decode(encoded, validate=True)
    return load_runs(BytesIO(payload)), hashlib.sha256(payload).hexdigest()


def observations(runs, selected=None):
    frames = []
    for number in selected if selected is not None else runs:
        run = runs[int(number)]
        frame = pd.DataFrame(run.observed, columns=NAMES)
        for key, values in run.additional.items():
            frame[key] = values
        frame.insert(0, "time_h", run.times)
        frame.insert(0, "run", int(number))
        frame["source_row"] = run.source_rows
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_summary(runs):
    return [
        {
            "run": int(n),
            "assays": len(r.times),
            "duration_h": len(r.events) - 1,
            "feed_A_mL": round(float(r.events[:, 1].sum() * 1000), 2),
            "feed_B_mL": round(float(r.events[:, 2].sum() * 1000), 2),
            "B_strength": r.strength,
            "final_volume_mL": round(r.volumes[-1] * 1000, 2),
        }
        for n, r in runs.items()
    ]


def replicate_analysis(runs, pair, parameters=None, nitrogen=None):
    if len(pair) != 2 or pair[0] == pair[1]:
        raise ValueError("Select two different replicate runs.")
    first, second = (runs[int(n)] for n in pair)
    common = np.intersect1d(first.times[1:], second.times[1:])
    if len(common) < 2:
        raise ValueError("At least two common post-initialization assay times are needed.")
    a = first.observed[np.searchsorted(first.times, common)]
    b = second.observed[np.searchsorted(second.times, common)]
    info = {
        "common_times_h": common.tolist(),
        "sd": np.sqrt(np.mean((a - b) ** 2, axis=0) / 2).tolist(),
    }
    if parameters is not None:
        predictions = {
            r.number: simulate_mechanistic(r, parameters, nitrogen) for r in (first, second)
        }
        info = repeatability(first, second, {n: p[:, : len(NAMES)] for n, p in predictions.items()})
    info.update(
        mean=((a + b) / 2).tolist(),
        first=a.tolist(),
        second=b.tolist(),
        adjusted=parameters is not None,
        adjusted_outputs=list(NAMES) if parameters is not None else [],
    )
    for column, name in enumerate(("GLN", "AMM"), len(NAMES)):
        if name in first.additional and name in second.additional:
            extra_a = first.additional[name][np.searchsorted(first.times, common)]
            extra_b = second.additional[name][np.searchsorted(second.times, common)]
            difference = extra_a - extra_b
            if parameters is not None and all(column < p.shape[1] for p in predictions.values()):
                difference -= (
                    predictions[first.number][common, column]
                    - predictions[second.number][common, column]
                )
                info["adjusted_outputs"].append(name)
            info["sd"].append(float(np.sqrt(np.mean(difference**2) / 2)))
            for key, array in (
                ("first", extra_a),
                ("second", extra_b),
                ("mean", (extra_a + extra_b) / 2),
            ):
                info[key] = np.column_stack((info[key], array)).tolist()
    return info


def recipe_from_run(run):
    start = int(run.times[0])
    return [
        dict(zip(RECIPE, [hour - start, *(run.events[hour] * 1000)], strict=True))
        for hour in range(start, len(run.events))
        if np.any(run.events[hour])
    ]


def scenario_defaults(run):
    return {
        "culture_age": float(run.times[0] + run.age_offset_h),
        "duration": len(run.events) - 1 - int(run.times[0]),
        "volume": round(float(initialize(run)[0] * 1000), 6),
        "strength": run.strength,
        **dict(zip(NAMES, run.observed[0].tolist(), strict=True)),
        **{k: float(v[0]) for k, v in run.additional.items()},
    }


def scenario_run(settings, rows):
    """Construct a new recipe with no future assay observations or state resets."""
    if any(settings.get(key) is None for key in ("duration", "volume", "strength", *NAMES)):
        raise ValueError(
            "Enter the duration, volume, feed strength and all initial concentrations."
        )
    duration = float(settings["duration"])
    if not duration.is_integer() or not 1 <= duration <= 500:
        raise ValueError("Duration must be a whole number from 1 to 500 hours.")
    age = float(settings.get("culture_age", 0))
    if not np.isfinite(age) or age < 0:
        raise ValueError("Initial culture age must be finite and nonnegative.")
    volume = float(settings["volume"]) / 1000
    initial = np.array([float(settings[name]) for name in NAMES])
    strength = float(settings["strength"])
    scales = np.array([1, float(settings.get("scale_A", 1)), float(settings.get("scale_B", 1)), 1])
    if not np.all(np.isfinite(np.r_[volume, initial, strength, scales])):
        raise ValueError("All operating conditions must be finite numbers.")
    if volume <= 0 or np.min(initial) < 0 or initial[1] < initial[0]:
        raise ValueError("Volume must be positive; concentrations nonnegative; TCD at least VCD.")
    if strength <= 0 or min(scales) < 0:
        raise ValueError("Feed B strength must be positive; feed scales cannot be negative.")
    events = recipe_events(int(duration), rows) * scales
    post_volumes = volume + np.cumsum(events[:, 1:].sum(axis=1) - events[:, 0])
    pre_volumes = np.r_[volume, post_volumes[:-1]]
    if np.any(events[:, 0] >= pre_volumes) or np.any(post_volumes <= 0):
        raise ValueError("A sample would empty the vessel. Reduce the sampling volume.")
    additional = {}
    for key in ("GLN", "AMM"):
        if key in settings:
            value = float(settings[key])
            if not np.isfinite(value) or value < 0:
                raise ValueError("Initial glutamine and ammonia must be finite and nonnegative.")
            additional[key] = np.array([value])
    return Run(
        0,
        events,
        post_volumes,
        np.array([0]),
        initial[None, :],
        (),
        strength,
        additional,
        age_offset_h=age,
    )


def recipe_events(duration, rows):
    events = np.zeros((duration + 1, 4))
    for row in rows:
        values = np.array([float(row[key] or 0) for key in RECIPE])
        hour = values[0]
        if not np.all(np.isfinite(values)) or min(values) < 0 or not hour.is_integer():
            raise ValueError(
                "Recipe entries must be finite and nonnegative, with whole-hour times."
            )
        if hour > duration:
            raise ValueError(f"Recipe hour {hour:g} exceeds the {duration}-hour simulation.")
        events[int(hour)] += values[1:] / 1000
    return events


def evaluate(runs, parameters):
    predictions = {n: simulate(r, parameters) for n, r in runs.items()}
    scores = [
        {"run": n, **dict(zip(NAMES, metrics(r, predictions[n])["rmse"], strict=True))}
        for n, r in runs.items()
    ]
    return predictions, scores


def parameter_rows(parameters, nitrogen=None):
    return [
        {"parameter": name, "value": round(value, 7)}
        for name, value in (
            asdict(parameters) | (asdict(nitrogen) if nitrogen is not None else {})
        ).items()
    ]

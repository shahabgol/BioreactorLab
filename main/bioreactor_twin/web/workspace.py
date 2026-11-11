"""Per-browser workspaces and bounded local training jobs; no shared mutable models."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from threading import RLock

import numpy as np

from bioreactor_twin.calibration.fit import fit
from bioreactor_twin.calibration.validation import (
    score_predictions,
    train_validated,
    validation_splits,
)
from bioreactor_twin.data.dataset import condition_key, load_runs
from bioreactor_twin.hybrid.training import HybridModel, fingerprint, train_hybrid
from bioreactor_twin.models.kinetics import Parameters
from bioreactor_twin.simulation.nitrogen import Nitrogen, simulate_mechanistic
from bioreactor_twin.web.services import decode_upload

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Workspace:
    runs: dict
    filename: str
    parameters: Parameters | None = None
    trained: tuple = ()
    label: str = "No trained model"
    predictions: dict = field(default_factory=dict)
    scores: list = field(default_factory=list)
    version: int = 0
    validation: dict = field(default_factory=dict)
    job: dict = field(default_factory=lambda: {"status": "idle"})
    hybrid: object = None
    bands: dict = field(default_factory=dict)
    invalid: dict = field(default_factory=dict)
    nitrogen: Nitrogen | None = None


class Lab:
    def __init__(self, root=ROOT):
        self.root = root
        self.sessions = {}
        self.lock = RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="twin-fit")

    def get(self, session):
        with self.lock:
            if session not in self.sessions:
                self.sessions[session] = self.example()
            return self.sessions[session]

    def example(self):
        sources = sorted((self.root / "data/raw").glob("*.xlsx"))
        if not sources:
            return Workspace({}, "No workbook loaded")
        result = Workspace(load_runs(sources[0]), sources[0].name)
        saved = self.root / "outputs/calibration/parameters.json"
        provenance = self.root / "outputs/calibration/provenance.json"
        matches = (
            provenance.exists()
            and json.loads(provenance.read_text()).get("source_sha256")
            == hashlib.sha256(sources[0].read_bytes()).hexdigest()
        )
        if saved.exists() and matches:
            result.parameters = Parameters(**json.loads(saved.read_text()))
            result.label = "Baseline · runs 7, 8"
            result.trained = (7, 8)
            extension = saved.with_name("nitrogen.json")
            if extension.exists():
                bundle = json.loads(extension.read_text())
                if (
                    bundle["source_sha256"] == json.loads(provenance.read_text())["source_sha256"]
                    and bundle["training_runs"] == list(result.trained)
                    and bundle["core"] == asdict(result.parameters)
                ):
                    result.nitrogen = Nitrogen(**bundle["nitrogen"])
                    result.validation = {"nitrogen": bundle["diagnostics"]}
            result.predictions = {
                n: simulate_mechanistic(r, result.parameters, result.nitrogen)
                for n, r in result.runs.items()
            }
            result.scores = score_predictions(result.runs, result.predictions)
        return result

    def import_data(self, session, contents=None, filename=None):
        current = self.get(session)
        if current.job["status"] in ("queued", "running"):
            raise ValueError("Wait for training to finish before replacing the dataset.")
        if contents is None:
            replacement = self.example()
        else:
            runs, _ = decode_upload(contents, filename)
            replacement = Workspace(runs, Path(filename).name)
        with self.lock:
            self.sessions[session] = replacement
        return replacement

    def set_replicates(self, session, numbers, independent=False):
        work = self.get(session)
        with self.lock:
            if work.job["status"] in ("queued", "running"):
                raise ValueError("Wait for training to finish before changing replicate groups.")
            chosen = set(numbers)
            if not chosen <= work.runs.keys() or len(chosen) < (1 if independent else 2):
                raise ValueError(
                    "Select at least two runs for a replicate group, or one to ungroup."
                )
            label = "Replicates " + ", ".join(map(str, sorted(chosen)))
            replacement = {
                n: replace(r, replicate_group=f"Condition {n}" if independent else label)
                if n in chosen
                else r
                for n, r in work.runs.items()
            }
            # Changing independence invalidates previous validation and trained closures.
            self.sessions[session] = Workspace(replacement, work.filename, version=work.version + 1)
        return "Replicate groups updated. Previous fit cleared; retrain using the new groups."

    def train(
        self,
        session,
        selected,
        starts,
        evaluations,
        mode="train_test",
        test_runs=(),
        model_type="first_principles",
        hybrid_options=None,
    ):
        work = self.get(session)
        selected = tuple(sorted(set(int(n) for n in selected)))
        if not selected or any(n not in work.runs for n in selected):
            raise ValueError("Select at least one available run for training.")
        if any(len(work.runs[n].times) < 2 for n in selected):
            raise ValueError("Each training run needs an initial assay and a later measurement.")
        if model_type not in ("first_principles", "hybrid"):
            raise ValueError("Choose a first-principles or hybrid model.")
        if model_type == "hybrid":
            from bioreactor_twin.hybrid.species import TARGETS, validate_targets

            validate_targets(work.runs, (hybrid_options or {}).get("targets", TARGETS))
        splits = validation_splits({n: work.runs[n] for n in selected}, mode, test_runs)
        if not 1 <= int(starts) <= 16 or not 10 <= int(evaluations) <= 1000:
            raise ValueError("Use 1–16 starts and 10–1000 evaluations per start.")
        with self.lock:
            if work.job["status"] in ("queued", "running"):
                raise ValueError("A training job is already active in this workspace.")
            stages = len(splits) + 1 if mode == "cross_validation" else 1
            work.job = {"status": "queued", "completed": 0, "starts": int(starts) * stages}
        self.executor.submit(
            self._fit,
            work,
            selected,
            int(starts),
            int(evaluations),
            mode,
            test_runs,
            model_type,
            hybrid_options,
        )
        return "Training queued. You can continue exploring your data."

    def _fit(
        self,
        work,
        selected,
        starts,
        evaluations,
        mode="train_test",
        test_runs=(),
        model_type="first_principles",
        hybrid_options=None,
    ):
        def progress(record):
            with self.lock:
                work.job.update(
                    completed=record["start"] + 1,
                    cost=record["cost"],
                    stage=record.get("stage", "Fitting"),
                )
                work.job["starts"] = max(work.job["starts"], record["start"] + 1)

        try:
            work.job["status"] = "running"
            result = (
                train_hybrid(
                    {n: work.runs[n] for n in selected},
                    mode,
                    test_runs,
                    starts,
                    evaluations,
                    hybrid_options,
                    progress,
                )
                if model_type == "hybrid"
                else train_validated(
                    {n: work.runs[n] for n in selected},
                    mode,
                    test_runs,
                    starts,
                    evaluations,
                    fit,
                    progress,
                )
            )
            with self.lock:
                work.parameters = result["parameters"]
                work.hybrid = result.get("model")
                work.nitrogen = result.get("nitrogen")
                work.bands = result.get("bands", {})
                work.invalid = result.get("invalid", {})
                work.predictions, work.scores = result["predictions"], result["scores"]
                work.trained = result["trained"]
                work.validation = result["validation"]
                converged = result["converged"]
                work.label = (
                    ("CV refit" if mode == "cross_validation" else "Refit")
                    + ("" if converged else " candidate")
                    + " · runs "
                    + ", ".join(map(str, work.trained))
                    + (" · Hybrid" if work.hybrid else "")
                )
                work.version += 1
                work.job.update(status="done", diagnostics=work.validation, converged=converged)
        except Exception as error:
            with self.lock:
                work.job.update(status="failed", error=str(error))

    def load_hybrid(self, session):
        work = self.get(session)
        if work.job["status"] in ("queued", "running"):
            raise ValueError("Wait for training to finish before loading a saved fit.")
        folder = self.root / "outputs/hybrid_training"
        model = HybridModel.restore(json.loads((folder / "model.json").read_text()))
        if model.fingerprint != fingerprint(work.runs):
            raise ValueError("Saved hybrid model does not match this dataset.")
        validation = json.loads((folder / "validation.json").read_text())
        from bioreactor_twin.hybrid.training import scores

        with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
            predictions = {int(n): saved[n].copy() for n in saved.files}
        bands = {}
        invalid = {}
        depletion = {}
        for n in predictions:
            producer = model
            if validation["mode"] == "cross_validation":
                producer = HybridModel.restore(
                    next(f["model"] for f in validation["folds"] if n in f["test_runs"])
                )
            depletion[n] = {}
            replay, bands[n], invalid[n] = producer.simulate(work.runs[n], diagnostics=depletion[n])
            expected = replay.copy()
            # Older bundles masked the entire column when the trajectory failed late.
            from bioreactor_twin.hybrid.species import OUTPUTS

            for name in invalid[n]:
                column = OUTPUTS.index(name)
                if np.all(np.isnan(predictions[n][:, column])):
                    expected[:, column] = np.nan
            if not np.allclose(expected, predictions[n], equal_nan=True):
                raise ValueError("Saved predictions do not match the exported fold models.")
            predictions[n] = replay
        with self.lock:
            work.hybrid = model
            work.nitrogen = None
            work.parameters = model.core
            validation["depletion_hours"] = {n: v for n, v in depletion.items() if v}
            validation["invalid_forecasts"] = {n: v for n, v in invalid.items() if v}
            work.validation = validation
            work.predictions = predictions
            work.bands = bands
            work.invalid = invalid
            work.trained = tuple(work.runs)
            work.scores = scores(work.runs, predictions, model.species)
            work.label = "Saved hybrid · CV predictions; all-data simulation refit"
            work.version += 1
            work.job = {"status": "idle"}
        return "Loaded saved hybrid fit. CV trajectories and final simulation model are available."

    def model_json(self, session):
        work = self.get(session)
        if work.parameters is None:
            raise ValueError("Train a model before exporting parameters.")
        return json.dumps(
            {
                "parameters": asdict(work.parameters),
                "training_runs": work.trained,
                "replicate_groups": {n: condition_key(r) for n, r in work.runs.items()},
                "model": work.label,
                "validation": work.validation,
                "hybrid": work.hybrid.export() if work.hybrid else None,
                "nitrogen": asdict(work.nitrogen) if work.nitrogen is not None else None,
            },
            indent=2,
        )

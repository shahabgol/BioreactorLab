"""Job completion/failure and session isolation, without long-running optimizations."""

import base64
import hashlib
import json
from dataclasses import replace

import pytest
from bioreactor_twin.models.kinetics import Parameters
from bioreactor_twin.simulation.nitrogen import Nitrogen
from bioreactor_twin.web.workspace import Lab, Workspace
from test_io_workflow import workbook
from test_numerics import synthetic_run


@pytest.fixture
def lab(tmp_path):
    result = Lab(tmp_path)
    result.sessions["one"] = Workspace({n: synthetic_run(n) for n in (1, 2, 7, 8)}, "test.xlsx")
    yield result
    result.executor.shutdown(wait=True)


def test_empty_workspace_and_parameter_export(lab):
    assert lab.get("new").runs == {}
    with pytest.raises(ValueError, match="Train"):
        lab.model_json("new")
    lab.get("one").parameters = Parameters()
    assert json.loads(lab.model_json("one"))["parameters"]["lysis"] == 0


def test_example_model_requires_source_fingerprint(tmp_path, lab):
    source = tmp_path / "data/raw"
    source.mkdir(parents=True)
    path = workbook(source)
    saved = tmp_path / "outputs/calibration"
    saved.mkdir(parents=True)
    from dataclasses import asdict

    (saved / "parameters.json").write_text(json.dumps(asdict(Parameters())))
    assert lab.example().parameters is None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (saved / "provenance.json").write_text(json.dumps({"source_sha256": digest}))
    assert lab.example().trained == (7, 8)
    extension = saved / "nitrogen.json"
    p = Nitrogen(0.4, 0.02, 0.05)
    bundle = {
        "source_sha256": digest,
        "training_runs": [7, 8],
        "core": asdict(Parameters()),
        "nitrogen": asdict(p),
        "diagnostics": {"success": True},
    }
    for change in ({"source_sha256": "different"}, {"training_runs": [1]}, {"core": {}}):
        extension.write_text(json.dumps(bundle | change))
        assert lab.example().nitrogen is None
    extension.write_text(json.dumps(bundle))
    work = lab.get("example")
    assert work.nitrogen == p
    assert "AMM" in work.scores[0]
    assert json.loads(lab.model_json("example"))["nitrogen"] == asdict(p)


def test_import_resets_model_and_does_not_change_another_session(tmp_path, lab):
    path = workbook(tmp_path)
    contents = "x," + base64.b64encode(path.read_bytes()).decode()
    uploaded = lab.import_data("new", contents, "new.xlsx")
    assert uploaded.parameters is None and list(uploaded.runs) == [7]
    assert len(lab.get("one").runs) == 4
    lab.get("new").job = {"status": "running"}
    with pytest.raises(ValueError, match="finish"):
        lab.import_data("new", contents, "new.xlsx")
    lab.get("new").job = {"status": "done"}
    assert lab.import_data("new").runs == {}


def test_training_validation_and_duplicate_job(lab):
    for ids, starts, evaluations in [([], 1, 20), ([99], 1, 20), ([1], 0, 20), ([1], 1, 1)]:
        with pytest.raises(ValueError):
            lab.train("one", ids, starts, evaluations)
    lab.get("one").job = {"status": "queued"}
    with pytest.raises(ValueError, match="active"):
        lab.train("one", [1], 1, 20)
    lab.get("one").job = {"status": "idle"}
    with pytest.raises(ValueError, match="no GLN"):
        lab.train("one", [1, 2, 7, 8], 1, 20, model_type="hybrid")
    lab.get("one").runs[1] = replace(synthetic_run(1), times=[])
    with pytest.raises(ValueError, match="later measurement"):
        lab.train("one", [1], 1, 20)


def test_background_training_publishes_only_complete_models(lab, monkeypatch):
    import bioreactor_twin.web.workspace as module

    def fake_fit(runs, **kwargs):
        assert [r.number for r in runs] == [1, 2]
        kwargs["on_progress"]({"start": 0, "cost": 1})
        return Parameters(), {"success": True}

    monkeypatch.setattr(module, "fit", fake_fit)

    class Immediate:
        def submit(self, fn, *args):
            fn(*args)

    executor = lab.executor
    lab.executor = Immediate()
    assert "queued" in lab.train("one", [2, 1], 1, 20)
    lab.executor = executor
    work = lab.get("one")
    assert work.job["status"] == "done" and work.trained == (1, 2)
    assert work.version == 1 and work.job["completed"] == 1
    monkeypatch.setattr(module, "fit", lambda *a, **k: (Parameters(), {"success": False}))
    lab._fit(work, (1,), 1, 20)
    assert "candidate" in work.label.lower()
    monkeypatch.setattr(
        module, "fit", lambda *a, **k: (_ for _ in ()).throw(ValueError("failed fit"))
    )
    lab._fit(work, (1,), 1, 20)
    assert work.job["status"] == "failed"
    assert work.parameters is not None  # previous successful model survives


def test_cross_validation_job_publishes_oof_results_and_final_model(lab, monkeypatch):
    import bioreactor_twin.web.workspace as module

    calls = []

    def fake_fit(runs, **kwargs):
        calls.append([r.number for r in runs])
        kwargs["on_progress"]({"start": 0, "cost": 0})
        return Parameters(), {"success": True}

    monkeypatch.setattr(module, "fit", fake_fit)

    class Immediate:
        def submit(self, fn, *args):
            fn(*args)

    executor = lab.executor
    lab.executor = Immediate()
    lab.train("one", [1, 2, 7, 8], 1, 20, mode="cross_validation")
    lab.executor = executor
    work = lab.get("one")
    assert work.job["starts"] == 4 and work.job["completed"] == 4
    assert work.label.startswith("CV refit")
    assert len(work.validation["folds"]) == 3
    assert calls[0] == [7, 8] and calls[-1] == [1, 2, 7, 8]
    assert json.loads(lab.model_json("one"))["validation"]["prediction_source"] == "out_of_fold"

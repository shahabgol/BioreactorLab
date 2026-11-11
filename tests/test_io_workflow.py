"""Synthetic workbook tests and end-to-end workflow outputs, without private files."""

import json
import runpy
import sys
from dataclasses import replace

import pytest
from bioreactor_twin.cli import main
from bioreactor_twin.data.audit import HEADERS, audit_workbook, summarize_records
from bioreactor_twin.data.dataset import load_runs
from bioreactor_twin.models.kinetics import Parameters
from openpyxl import Workbook
from test_numerics import synthetic_run


def workbook(tmp_path, *, sheet="SampleData", header=HEADERS, times=(0, 1)):
    book = Workbook()
    tab = book.active
    tab.title = sheet
    tab.append(header)
    for time in times:
        tab.append([7, time, 50000, 350000, 300000, 6, 0, 0, 0, 199, 0, 7, 7, 0.02, 0.5, 0.2, 0.01])
    path = tmp_path / "synthetic.xlsx"
    book.save(path)
    book.close()
    return path


def test_readers_and_audit_cli(tmp_path):
    path = workbook(tmp_path)
    result = audit_workbook(path)
    assert result["row_count"] == 2
    run = load_runs(path)[7]
    assert run.observed[0, 0] == 0.3
    assert run.source_rows == (2, 3)
    output = tmp_path / "nested" / "audit.json"
    assert main(["audit", str(path), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["row_count"] == 2


@pytest.mark.parametrize("reader", [audit_workbook, load_runs])
def test_wrong_headers_raise_and_close(tmp_path, reader):
    with pytest.raises(ValueError, match="columns"):
        reader(workbook(tmp_path, header=("bad",) * 17))


def test_sheet_error_and_nonhourly_ledger(tmp_path):
    with pytest.raises(ValueError, match="sheets"):
        audit_workbook(workbook(tmp_path, sheet="Wrong"))
    with pytest.raises(ValueError, match="hourly"):
        load_runs(workbook(tmp_path, times=(0, 2)))


@pytest.mark.parametrize("value", ["=1+1", "#N/A"])
def test_audit_rejects_formulas_and_excel_errors(tmp_path, value):
    from openpyxl import load_workbook

    path = workbook(tmp_path)
    book = load_workbook(path)
    book.active["F2"] = value
    book.save(path)
    book.close()
    with pytest.raises(ValueError, match="Formula or Excel error"):
        audit_workbook(path)


def test_audit_skips_blank_rows(tmp_path):
    from openpyxl import load_workbook

    path = workbook(tmp_path)
    book = load_workbook(path)
    book.active["A5"] = None
    book.active["A6"] = 7
    for column in range(2, 18):
        book.active.cell(6, column, book.active.cell(3, column).value)
    book.active["B6"] = 2
    book.save(path)
    book.close()
    assert audit_workbook(path)["row_count"] == 3


def test_audit_structural_edge_cases():
    from test_source_audit import record

    with pytest.raises(ValueError, match="no experimental"):
        summarize_records([])
    row = record(0.5, 199)
    with pytest.raises(ValueError, match="integers"):
        summarize_records([row])
    with pytest.raises(ValueError, match="positive"):
        summarize_records([record(0, 0)])


def test_workflow_holds_out_pair_and_writes_real_plots(tmp_path, monkeypatch):
    import bioreactor_twin.workflow as workflow

    runs = {i: replace(synthetic_run(i), strength=0.5 if i == 8 else 1) for i in (1, 2, 7, 8)}
    monkeypatch.setattr(workflow, "load_runs", lambda path: runs)

    def fitted(training, **kwargs):
        assert [run.number for run in training] == [7, 8]
        return Parameters(), {"success": True}

    monkeypatch.setattr(workflow, "fit", fitted)
    assert main(["fit", "unused.xlsx", "--output", str(tmp_path)]) == 0
    assert len(list(tmp_path.glob("*_comparison.png"))) == 5
    assert len(list(tmp_path.glob("run_*_hourly.csv"))) == 4
    assert json.loads((tmp_path / "parameters.json").read_text())["k_g"] == 3
    checks = json.loads((tmp_path / "numerical_checks.json").read_text())
    assert checks["1"]["1.0"] < 1e-7


def test_module_entry_point(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bioreactor_twin"])
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("bioreactor_twin", run_name="__main__")
    assert caught.value.code == 0

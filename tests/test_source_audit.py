"""Check the audit rules that materially affect model development."""

import pytest
from bioreactor_twin.cli import main
from bioreactor_twin.data.audit import (
    DCD,
    GLC,
    GLN,
    GLU,
    HEADERS,
    LAC,
    NH3,
    RUN,
    SAMPLE,
    TCD,
    TIME,
    VCD,
    VOLUME,
    A,
    B,
    C,
    summarize_records,
)


def record(time, volume, *, sample=0, feed_a=0):
    result = dict.fromkeys(HEADERS)
    result.update({RUN: 1, TIME: time, VOLUME: volume, SAMPLE: sample, A: feed_a, B: 0, C: 0})
    return result


def test_ledger_uses_current_events_and_does_not_reapply_initial_feed():
    initial = record(0, 199)
    initial[C] = 40
    sample = record(1, 196.32, sample=2.68)
    fed = record(2, 201.32, feed_a=5)
    run = summarize_records([initial, sample, fed])["runs"][0]
    assert run["max_abs_volume_ledger_error_after_initial_row_mL"] < 1e-10


def test_volume_mismatch_is_reported_without_repair():
    rows = [record(0, 199), record(1, 198, sample=2)]
    result = summarize_records(rows)["runs"][0]
    assert result["volume_ledger_discrepancies_over_1e-6_mL"] == [{"time_h": 1, "residual_mL": 1}]
    assert rows[1][VOLUME] == 198


def test_repeated_panel_flag_retains_both_records_and_sampling_events():
    rows = [record(39, 199), record(42, 197.64, sample=1.36)]
    for row in rows:
        row.update({VCD: 1e6, DCD: 1e5, TCD: 1.1e6, GLC: 5, GLN: 0, GLU: 0.5, LAC: 1, NH3: 0})
    result = summarize_records(rows)["runs"][0]
    assert result["identical_measurement_panels_at_h"] == [[39, 42]]
    assert result["measurement_panels"] == 2
    assert result["total_events_mL_including_initial_row"][SAMPLE] == 1.36
    assert result["zero_measurements"][GLN] == [39, 42]


def test_blank_measurements_are_not_interpolated_or_counted():
    result = summarize_records([record(0, 199), record(1, 199)])["runs"][0]
    assert result["measurement_panels"] == 0
    assert result["max_abs_cell_identity_error_cells_per_mL"] is None


def test_duplicate_keys_are_rejected():
    with pytest.raises(ValueError, match="Duplicate run/time"):
        summarize_records([record(0, 199), record(0, 199)])


@pytest.mark.parametrize("bad", [None, "2", -1, float("nan"), True])
def test_invalid_event_values_are_rejected(bad):
    row = record(0, 199)
    row[SAMPLE] = bad
    with pytest.raises(ValueError):
        summarize_records([row])


def test_cell_count_identity_is_checked_independently():
    row = record(0, 199)
    row.update({VCD: 100, DCD: 20, TCD: 125})
    assert summarize_records([row])["runs"][0]["max_abs_cell_identity_error_cells_per_mL"] == 5


def test_cli_explains_available_commands(capsys):
    assert main([]) == 0
    assert "use audit or fit" in capsys.readouterr().out

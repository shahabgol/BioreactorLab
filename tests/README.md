# Test strategy

Implemented tests check source audit behavior using synthetic records, including the hour-zero boundary, volume mismatch, duplicate assay flags, sampling retention, missing observations, duplicate keys, invalid event values and the dependent cell-count identity. The local-data test verifies the actual source fingerprints and expected inventory; it skips when private sources are absent.

`test_numerics.py` checks analytical growth/death, feed/sample conservation, zero-substrate boundaries, physical parameters/states, refinement/reference agreement and event timing. `test_calibration.py` checks synthetic fitting, residual failures, uncertainty arithmetic and metrics. `test_io_workflow.py` checks workbook input and command-line exports. `test_control.py` covers MPC and disturbance adaptation.

Run `python -m pytest --cov=bioreactor_twin --cov-branch --cov-fail-under=100`. All executable application statements and measured branches are currently exercised; CI requires the same coverage using synthetic fixtures alone. Coverage is a check on tested execution paths, not proof of correctness for all inputs or of biological accuracy.

Use closed-form solutions or independent balances as expected results. Do not test only against another call to the same implementation. Calibration accuracy thresholds are scientific validation criteria, separate from source-audit test success. CI must never need the original workbook.

# Workbook format

The input is an `.xlsx` workbook with a sheet named `SampleData`. Generate an independent
example with `python scripts/make_demo_workbook.py`. The loader expects the following columns
in this order:

```text
Vessel
Feed_Elapsed_Time (h)
DCC (cells/mL)
TCC (cells/mL)
VCC (cells/mL)
GLC (g/L)
Feed_Vsampled mL
Feed_A mL
Feed_B mL
Vessel_Volume_Calculated_mL
Feed_C mL
pH measured
pH temp corrected
GLN (g/L)
GLU (g/L)
LAC (g/L)
NH3 (g/L)
```

Use one row per hour per vessel, starting at hour zero. Report event amounts in mL and
the vessel volume after those events. Assays correspond to the state before same-hour
sampling and feeding. Leave unmeasured assay fields blank. Use numeric values for
reported measurements and event amounts. The first assay initializes the model.

The demo uses vessel IDs 101–103. The loader preserves compatibility with one legacy
study by grouping IDs 1/2 by default, applying a half-strength Feed B rule to ID 8 and
excluding the assay at hour 42 for ID 7 while retaining the sample withdrawal. Use
different IDs for unrelated studies. Replicate groups can be edited in the app.

The simulator uses synthetic default feed concentrations unless a local
`configs/development.yaml` or the path set in `BIOREACTOR_FEED_CONFIG` supplies a feed
recipe. The supported keys below are in g/L. Set them to match your process before
training and keep proprietary formulations in ignored local files.

```yaml
feeds_g_per_L:
  A: {GLC: 400}
  B: {GLC: 50, GLU: 2, GLN: 0.02}
```

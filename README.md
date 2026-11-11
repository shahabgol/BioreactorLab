# BioreactorLab

[![CI](https://github.com/shahabgol/BioreactorLab/actions/workflows/ci.yml/badge.svg)](https://github.com/shahabgol/BioreactorLab/actions/workflows/ci.yml)

BioreactorLab is a local Dash app and Python package for simulating a fed-batch bioreactor, fitting kinetic models, and comparing glucose-feed control and disturbance adaptation in simulation.

- **First principles** uses cell and solute balances, hourly RK4 integration, and exact sampling and bolus updates.
- **Hybrid** keeps the mechanistic cell, glucose, and glutamate core and learns integrated lactate, glutamine, and ammonia rates.
- **Validation** holds out whole runs or replicate groups and plots predictions against assays at their recorded times.
- **Simulation** compares feed recipes. The first-principles control experiments include model predictive control and moving-horizon estimation for a synthetic growth disturbance.

## Run the app

Use Python 3.12. From the repository root:

```sh
python -m venv .venv
```

Activate the environment using `source .venv/bin/activate` on macOS/Linux or `.\.venv\Scripts\Activate.ps1` in PowerShell. Then run:

```sh
python -m pip install -r environment/requirements.lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python scripts/make_demo_workbook.py
bioreactor-lab
```

Open [localhost:8050](http://localhost:8050). The generated demo contains independent synthetic batches and is ignored by Git. In **Model training**, select the first-principles or hybrid model and choose grouped cross-validation or a whole-run holdout. Run the fit before opening **Simulation**. To use your own data, import a workbook that matches the [input schema](docs/WORKBOOK_FORMAT.md) and configure your feed concentrations locally.

The app binds to the loopback address. Uploaded data and active fits live in the local server's memory. Use the export buttons to retain results between sessions. The control experiments are synthetic demonstrations.

## Development

```sh
python -m ruff check .
python -m ruff format --check .
python -m pytest -m "not local_data" --cov=bioreactor_twin --cov-branch --cov-fail-under=100
python -m build --no-isolation
```

GitHub Actions checks the tracked file set, dependencies, lint, formatting, synthetic tests and package build on Ubuntu and Windows. A separate manually triggered workflow builds downloadable release candidates.

Keep study workbooks, private documents, local feed recipes, calibrated models and generated results outside the tracked tree. The public repository contains code and synthetic fixtures. The [MIT license](LICENSE) applies to the published software.

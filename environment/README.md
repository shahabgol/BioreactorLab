# Python environment

Use Python 3.12 and a project-local `.venv`. Runtime dependency ranges and development extras are declared in `pyproject.toml`; the resolved development environment is pinned, including transitive dependency versions, in `requirements.lock.txt`.

PowerShell: run `./environment/bootstrap.ps1`. If `python` is a Windows Store alias, pass a real interpreter: `./environment/bootstrap.ps1 -Python 'C:/path/to/python.exe'`.

Linux/macOS:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r environment/requirements.lock.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m build --no-isolation
```

Refresh the lock deliberately after changing dependencies:

```sh
python -m piptools compile --extra dev --allow-unsafe --strip-extras --output-file environment/requirements.lock.txt pyproject.toml
```

The lock targets CPython 3.12 and includes Dash and Plotly. CI exercises both Windows and Ubuntu with the same lock. Local checks do not establish that the remote jobs have run. No cloud keys or secrets are needed. The local app runs with `python -m bioreactor_twin.web.app`. A production WSGI server, authentication and cloud hosting remain future deployment work.

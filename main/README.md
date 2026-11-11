# Application package

`main/bioreactor_twin` is the installable Python package:

| Package | Responsibility |
|---|---|
| data | Provenance, unit conversion, observation/event/run tables and overrides |
| models | Pure rate laws, parameter units, amount balances and observation operator |
| simulation | Fixed one-hour RK4, exact sample/feed jumps and event-side output |
| calibration | Standardized residuals, bounded multi-start fitting and Jacobian diagnostics |
| validation | Grouped whole-run evaluation, residuals and uncertainty |
| hybrid | Integrated metabolite rates and grouped feature selection |
| control | Bounded glucose boluses, moving-horizon adaptation and synthetic experiments |
| web | Dash layout/callbacks, Plotly views, isolated workspaces, training jobs and scenario services |

Keep numerical equations independent of file I/O, fitting and plots. Prefer explicit state/parameter names over positional arrays at public interfaces. Convert units at the data boundary and document state-array order centrally. Record numerical failures with run, time, state, parameters and last event.

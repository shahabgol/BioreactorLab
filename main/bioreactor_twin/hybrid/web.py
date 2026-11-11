"""Hybrid review views; plotting uses the fold that produced each prediction."""

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update

from bioreactor_twin.hybrid.core import trajectory
from bioreactor_twin.hybrid.regressors import RateModel
from bioreactor_twin.hybrid.species import OUTPUTS
from bioreactor_twin.models.kinetics import Parameters
from bioreactor_twin.web.charts import COLORS, measured
from bioreactor_twin.web.layout import table


def diagnostics(work, selected, output, training=False):
    index = OUTPUTS.index(output)
    cv = work.validation.get("mode") == "cross_validation"
    parity = go.Figure()
    residual = go.Figure()
    residual_prediction = go.Figure()
    rows = []
    endpoints = []
    for n in selected or []:
        if n not in work.predictions or (not cv and n in work.trained and not training):
            continue
        run = work.runs[n]
        predicted = work.predictions[n][run.times[1:], index]
        actual = measured(run, output)[1:]
        valid = np.isfinite(actual) & np.isfinite(predicted)
        if not np.all(valid):
            rows.append(
                dict(run=n, RMSE=None, MAE=None, bias=None, R2=None, status="Invalid trajectory")
            )
            continue
        error = predicted - actual
        variance = np.sum((actual - np.mean(actual)) ** 2)
        rows.append(
            dict(
                run=n,
                RMSE=float(np.sqrt(np.mean(error**2))),
                MAE=float(np.mean(abs(error))),
                bias=float(np.mean(error)),
                R2=float(1 - np.sum(error**2) / variance) if variance > 1e-15 else None,
                status="CV held out" if cv else ("Training" if n in work.trained else "Held out"),
            )
        )
        color = COLORS[list(work.runs).index(n) % 4]
        parity.add_trace(
            go.Scatter(
                x=actual,
                y=predicted,
                mode="markers",
                name=f"Run {n}",
                customdata=run.times[1:],
                marker=dict(color=color),
                hovertemplate=(
                    "Measured %{x:.4g}; predicted %{y:.4g}"
                    "<br>Hour %{customdata}<extra>%{fullData.name}</extra>"
                ),
            )
        )
        residual.add_trace(
            go.Scatter(
                x=run.times[1:], y=error, mode="markers", name=f"Run {n}", marker=dict(color=color)
            )
        )
        endpoints.extend([*actual, *predicted])
        residual_prediction.add_trace(
            go.Scatter(
                x=predicted, y=error, mode="markers", name=f"Run {n}", marker=dict(color=color)
            )
        )
    if endpoints:
        low, high = min(endpoints), max(endpoints)
        pad = max((high - low) * 0.05, 0.001)
        limits = [low - pad, high + pad]
        parity.add_trace(
            go.Scatter(
                x=limits, y=limits, mode="lines", name="y = x", line=dict(color="gray", dash="dot")
            )
        )
        parity.update_xaxes(range=limits)
        parity.update_yaxes(range=limits, scaleanchor="x", scaleratio=1)
    else:
        parity.add_annotation(
            text="No eligible held-out points. Enable training points for calibration diagnostics.",
            showarrow=False,
        )
    parity.update_layout(
        template="plotly_white",
        height=420,
        xaxis_title=f"Measured {output}",
        yaxis_title=f"Predicted {output}",
    )
    residual.update_layout(
        template="plotly_white",
        height=300,
        xaxis_title="Actual assay hour",
        yaxis_title="Prediction minus measurement",
    )
    residual.add_hline(y=0, line_dash="dot")
    residual_prediction.update_layout(
        template="plotly_white",
        height=300,
        xaxis_title=f"Predicted {output}",
        yaxis_title="Residual",
    )
    residual_prediction.add_hline(y=0, line_dash="dot")
    parity.update_xaxes(constrain="domain")
    parity.update_yaxes(constrain="domain")
    for row in rows:
        for key in ("RMSE", "MAE", "bias", "R2"):
            if row[key] is not None:
                row[key] = float(f"{row[key]:.4g}")
    return [
        html.P("Selected runs; initial assays excluded. Cells: million/mL; metabolites: g/L."),
        dcc.Graph(figure=parity),
        dcc.Graph(figure=residual),
        dcc.Graph(figure=residual_prediction),
        table(
            "hybrid-diagnostic-table",
            [(k, k) for k in ("run", "status", "RMSE", "MAE", "bias", "R2")],
            **{},
        ),
    ], rows


def review(work, view, selected, output, training=False):
    if work.hybrid is None:
        return html.P("Train a hybrid model to review its regressions.", className="result-note")
    if view == "diagnostics":
        children, rows = diagnostics(work, selected, output, training)
        children[-1].data = rows
        return html.Div(children)
    species = output if output in work.hybrid.species else next(iter(work.hybrid.species))
    if view == "features":
        report = work.hybrid.selection[species]
        rows = [
            {
                **x,
                "features": ", ".join(x["features"]),
                "rmse": float(f"{x['rmse']:.4g}") if x["rmse"] is not None else None,
                "fold_errors": ", ".join(f"{v:.4g}" for v in x["fold_errors"]),
            }
            for x in report["candidates"]
        ]
        result = table(
            "hybrid-feature-table",
            [(k, k) for k in ("family", "features", "penalty", "length", "rmse", "fold_errors")],
        )
        result.data = rows
        counts = {
            name: sum(
                name in fold["model"]["selection"][species]["selected"]["features"]
                for fold in work.validation["folds"]
            )
            for name in ("VCD", "viability", "GLC", "GLU", "mu", "time")
        }
        frequency = go.Figure(go.Bar(x=list(counts), y=list(counts.values())))
        frequency.update_layout(
            template="plotly_white",
            height=260,
            title="Feature selection across validation folds (descriptive)",
            yaxis_title="Number of folds",
            yaxis=dict(dtick=1),
        )
        fold_table = table("hybrid-fold-features", [(k, k) for k in ("fold", "family", "features")])
        fold_table.data = [
            dict(
                fold=i + 1,
                family=f["model"]["selection"][species]["selected"]["family"],
                features=", ".join(f["model"]["selection"][species]["selected"]["features"])
                or "Constant rate",
            )
            for i, f in enumerate(work.validation["folds"])
        ]
        selected = report["selected"]
        return html.Div(
            [
                html.P(
                    f"{species}: {selected['family'].upper()} using "
                    f"{', '.join(selected['features']) or 'a constant rate'}. {report['note']}"
                ),
                html.P(
                    "GP: 16-landmark Matérn-5/2 integral approximation. "
                    "Selection frequencies across CV folds are descriptive."
                ),
                fold_table,
                dcc.Graph(figure=frequency),
                result,
            ]
        )
    figure = go.Figure()
    for n in selected or []:
        model = work.hybrid.species[species]
        parameters = work.parameters
        if work.validation.get("mode") == "cross_validation":
            fold = next(f for f in work.validation["folds"] if n in f["test_runs"])
            model = RateModel(**fold["model"]["species"][species])
            parameters = Parameters(**fold["model"]["core"])
        trace = trajectory(work.runs[n], parameters)
        figure.add_trace(
            go.Scatter(x=trace.nodes[:, 5], y=model.rates(trace), name=f"Run {n}", mode="lines")
        )
    figure.update_layout(
        template="plotly_white",
        height=400,
        xaxis_title="Recorded hour",
        yaxis_title=f"{species} signed net rate (g/billion cells/h)",
    )
    return html.Div(
        [
            html.P(
                "Signed rates: production positive, consumption negative. No direct rate assays."
            ),
            dcc.Graph(figure=figure),
        ]
    )


def register_hybrid(app, lab):
    @app.callback(
        Output("saved-hybrid-status", "children"),
        Output("model-type", "value"),
        Input("load-hybrid", "n_clicks"),
        State("session", "data"),
        prevent_initial_call=True,
    )
    def load(clicks, session):
        try:
            return lab.load_hybrid(session), "hybrid"
        except (ValueError, OSError, KeyError) as error:
            return f"Could not load hybrid fit: {error}", no_update

    @app.callback(Output("hybrid-controls", "style"), Input("model-type", "value"))
    def controls(kind):
        return {} if kind == "hybrid" else {"display": "none"}

    @app.callback(
        Output("initial-GLN", "value"),
        Output("initial-AMM", "value"),
        Output("initial-age", "value"),
        Input("reference-run", "value"),
        Input("load-recipe", "n_clicks"),
        State("session", "data"),
    )
    def initial(number, clicks, session):
        run = lab.get(session).runs.get(number)
        return tuple(
            float(run.additional[k][0]) if run and k in run.additional else 0.0
            for k in ("GLN", "AMM")
        ) + (float(run.times[0] + run.age_offset_h) if run else 0.0,)

    @app.callback(
        Output("hybrid-review", "children"),
        Input("model-version", "data"),
        Input("dataset-version", "data"),
        Input("hybrid-view", "value"),
        Input("plot-runs", "value"),
        Input("output", "value"),
        Input("diagnostic-training", "value"),
        State("session", "data"),
    )
    def show(version, revision, view, selected, output, training, session):
        return review(lab.get(session), view, selected, output, bool(training))

    return {
        "hybrid_controls": controls,
        "hybrid_initial": initial,
        "hybrid_review": show,
        "hybrid_load": load,
    }

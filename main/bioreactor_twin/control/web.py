"""Task 2 controls and review plots, layered onto the existing recipe editor."""

import json
from dataclasses import fields

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update
from plotly.subplots import make_subplots

from bioreactor_twin.control.dynamics import Settings, reference_target, target_profile
from bioreactor_twin.control.experiments import experiment
from bioreactor_twin.web.services import NAMES, scenario_run

DEFAULTS = dict(
    horizon=48,
    interval=6,
    assay_interval=24,
    delay=0,
    window=72,
    disturbance=72,
    multiplier=0.5,
    max_dose_ml=2,
    max_volume_ml=300,
    max_glucose=15,
    noise=0,
    seed=42,
)


def controls():
    from bioreactor_twin.web.layout import action, field, number, table

    grid = table("control-target", [("hour", "Hour"), ("VCD", "Target VCD")], editable=True)
    grid.data = [
        dict(hour=h, VCD=v) for h, v in ((0, 0.3), (72, 2), (120, 8), (200, 15), (500, 15))
    ]
    return html.Div(
        [
            field(
                "Experiment",
                dcc.Dropdown(
                    id="task2-mode",
                    value="recipe",
                    clearable=False,
                    options=[
                        dict(label=a, value=b)
                        for a, b in [
                            ("Recipe simulation", "recipe"),
                            ("VCD control", "control"),
                            ("Disturbance & adaptation", "adaptation"),
                        ]
                    ],
                    className="light-select",
                ),
            ),
            html.Div(
                [
                    html.P(
                        "Synthetic experiment with the current first-principles model. "
                        "Shared conditions/recipe below apply. MPC replaces Feed A.",
                        className="side-note",
                    ),
                    *[
                        field(
                            name.replace("_", " "),
                            number(
                                "task2-" + name,
                                value,
                                0,
                                None,
                                1
                                if name
                                in (
                                    "horizon",
                                    "interval",
                                    "assay_interval",
                                    "delay",
                                    "window",
                                    "disturbance",
                                    "seed",
                                )
                                else 0.1,
                            ),
                        )
                        for name, value in DEFAULTS.items()
                    ],
                    field(
                        "VCD target source",
                        dcc.Dropdown(
                            id="task2-target-source",
                            value="reference",
                            clearable=False,
                            options=[
                                dict(label="Constrained reference recipe", value="reference"),
                                dict(label="Custom VCD breakpoints", value="custom"),
                            ],
                            className="light-select",
                        ),
                    ),
                    html.P(
                        "Custom VCD target (control mode only). Hour zero must match initial VCD; "
                        "Last hour must cover duration. Edit the target breakpoints."
                    ),
                    grid,
                    action("task2-run", "Run Task 2 experiment"),
                    html.Div(id="task2-status", role="status", className="side-note"),
                    action("task2-export", "Export experiment JSON", True),
                ],
                id="task2-controls",
                style={"display": "none"},
            ),
            dcc.Store(id="task2-result"),
            dcc.Download(id="task2-download"),
        ]
    )


def feed_traces(name, feed_ml, color, symbol):
    """Use stems for instantaneous doses and markers for exact dose hover."""
    doses = np.asarray(feed_ml)
    hours = np.flatnonzero(doses > 0)
    amounts = doses[hours]
    group = f"feed:{name}"
    return (
        go.Scatter(
            x=[x for hour in hours for x in (int(hour), int(hour), None)],
            y=[y for amount in amounts for y in (0.0, float(amount), None)],
            mode="lines",
            line=dict(color=color, width=2),
            legendgroup=group,
            showlegend=False,
            hoverinfo="skip",
        ),
        go.Scatter(
            x=hours,
            y=amounts,
            mode="markers",
            marker=dict(color=color, symbol=symbol, size=9),
            name=name,
            legendgroup=group,
            hovertemplate="Hour %{x}<br>Feed A: %{y:.3f} mL<extra>%{fullData.name}</extra>",
        ),
    )


def figure(result):
    from bioreactor_twin.web.charts import COLORS

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.135,
        subplot_titles=(
            "VCD",
            "Glucose (g/L)",
            "Glucose feed A boluses (mL)",
            "Kinetic multipliers",
        ),
    )
    symbols = ("circle", "diamond-open", "square-open", "x")

    for index, (name, item) in enumerate(result["cases"].items()):
        values = np.array(item["truth"])
        hours = np.arange(len(values))
        color = COLORS[index % len(COLORS)]
        fig.add_trace(
            go.Scatter(x=hours, y=values[:, 0], name=name, line=dict(color=color)), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=hours, y=values[:, 2], name=name, line=dict(color=color)),
            row=2,
            col=1,
        )
        for trace in feed_traces(name, item["feed_ml"], color, symbols[index % len(symbols)]):
            fig.add_trace(trace, row=3, col=1)
        if result["mode"] == "adaptation":
            for model, estimates in item["estimates"].items():
                fig.add_trace(
                    go.Scatter(
                        x=hours,
                        y=np.array(estimates)[:, 0],
                        name=model + " estimate",
                        line=dict(dash="dash"),
                    ),
                    row=1,
                    col=1,
                )
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=item["factors"],
                name=name + " · estimated growth",
                line=dict(color=color),
                line_shape="hv",
            ),
            row=4,
            col=1,
        )
        if "maintenance_factors" in item:
            fig.add_trace(
                go.Scatter(
                    x=hours,
                    y=item["maintenance_factors"],
                    name=name + " · maintenance",
                    line=dict(color=color, dash="dash"),
                    line_shape="hv",
                ),
                row=4,
                col=1,
            )
        if name != "Recorded recipe" and name != "Nominal MPC":
            fig.add_trace(
                go.Scatter(
                    x=hours,
                    y=np.where(
                        hours >= item["settings"]["disturbance"], item["settings"]["multiplier"], 1
                    ),
                    name=name + " · true growth",
                    line=dict(color=color, dash="dot"),
                ),
                row=4,
                col=1,
            )
    if result["target"] is not None:
        fig.add_trace(
            go.Scatter(
                y=result["target"], name="VCD target", line=dict(color="black", dash="dash")
            ),
            row=1,
            col=1,
        )
    for row, title in enumerate(fig.layout.annotations, start=1):
        legend = "legend" if row == 1 else f"legend{row}"
        top = fig.get_subplot(row, 1).yaxis.domain[1]
        fig.update_traces(legend=legend, row=row, col=1)
        fig.update_layout(
            **{
                legend: dict(
                    x=0,
                    y=top + 0.008,
                    xanchor="left",
                    yanchor="bottom",
                    orientation="h",
                    entrywidthmode="fraction",
                    entrywidth=0.5,
                    font=dict(size=11),
                    itemclick="toggle",
                    itemdoubleclick="toggleothers",
                    groupclick="togglegroup",
                )
            }
        )
        title.update(x=0, xanchor="left", y=top + 0.09)
    fig.update_layout(
        template="plotly_white",
        height=1660,
        margin=dict(t=164, r=24, b=60, l=58),
    )
    fig.update_xaxes(
        title_text="Scenario hour",
        title_standoff=6,
        showticklabels=True,
        showline=True,
        linecolor="#d8e4ef",
        ticks="outside",
        ticklen=3,
    )
    fig.update_yaxes(rangemode="tozero", row=3, col=1)
    return fig


def review(result):
    from bioreactor_twin.web.layout import table

    if not result:
        return html.P(
            "Run an experiment to compare control or adaptation strategies.",
            className="task2-note",
        )
    rows = []
    for name, item in result["cases"].items():
        if result["mode"] == "control":
            rows.append(
                dict(
                    case=name,
                    VCD_RMSE=round(item["tracking_rmse"], 4),
                    feed_ml=round(sum(item["feed_ml"]), 3),
                    failed_decisions=sum(not r["success"] for r in item["controller"]),
                    violations=len(item["violations"]),
                    unreachable_hours=sum(not r["feasible"] for r in item["reachable"]),
                )
            )
        else:
            for score in item["scores"]:
                rows.append(
                    dict(
                        model=score["model"],
                        lead_h=score["lead_h"] if score["lead_h"] is not None else "Next assay",
                        count=score["count"],
                        VCD_RMSE=round(score["rmse"][0], 4) if score["rmse"] else None,
                        glucose_RMSE=round(score["rmse"][2], 4) if score["rmse"] else None,
                    )
                )
    grid = table("task2-metrics", [(k, k) for k in rows[0]])
    grid.data = rows
    return html.Div(
        [
            html.P(result["note"], className="task2-note"),
            html.P(
                "Daily assay corrections are not hourly measurements. Forecast scores use only "
                "forecasts issued after disturbance onset and before future assays arrive. "
                "Oracle knows the true process and is an ideal benchmark. "
                "Failed MPC decisions apply zero Feed A; fixed B/C can still violate limits.",
                className="task2-note",
            ),
            dcc.Graph(figure=figure(result)),
            grid,
            html.P(
                "Export JSON to inspect forecasts, estimator convergence/sensitivity, "
                "feeds, constraints and settings. Uncertainty coverage is not established.",
                className="task2-note",
            ),
        ]
    )


def register(app, lab):
    @app.callback(
        Output("task2-controls", "style"),
        Output("task2-results", "style"),
        Input("task2-mode", "value"),
    )
    def visibility(mode):
        style = {"display": "none"} if mode == "recipe" else {}
        return style, style

    @app.callback(
        Output("task2-result", "data"),
        Output("task2-status", "children"),
        Input("task2-run", "n_clicks"),
        Input("model-version", "data"),
        Input("dataset-version", "data"),
        State("task2-mode", "value"),
        State("session", "data"),
        *[State("task2-" + f.name, "value") for f in fields(Settings)],
        State("control-target", "data"),
        State("recipe-table", "data"),
        *[
            State(key, "value")
            for key in (
                "duration",
                "volume",
                "strength",
                "scale-a",
                "scale-b",
                *["initial-" + n for n in NAMES],
            )
        ],
        State("task2-target-source", "value"),
        prevent_initial_call=True,
        running=[(Output("task2-run", "disabled"), True, False)],
    )
    def run(clicks, model_version, dataset_version, mode, session, *values):
        from dash import ctx

        if ctx.triggered_id != "task2-run":
            return None, "Model or dataset changed; rerun the experiment."
        try:
            work = lab.get(session)
            if work.parameters is None or work.hybrid is not None:
                raise ValueError("Load/train a first-principles model before Task 2 experiments.")
            settings = Settings(**dict(zip(DEFAULTS, values[:12], strict=True)))
            settings.validate()
            targets, recipe, *operating = values[12:]
            duration, volume, strength, scale_a, scale_b, *initial = operating[:10]
            source = next(iter(operating[10:]), "custom")
            scenario = scenario_run(
                dict(
                    duration=duration,
                    volume=volume,
                    strength=strength,
                    scale_A=scale_a,
                    scale_B=scale_b,
                    **dict(zip(NAMES, initial, strict=True)),
                ),
                recipe,
            )
            target = None
            if mode == "control":
                target = (
                    reference_target(scenario, work.parameters, settings)
                    if source == "reference"
                    else target_profile(targets, int(duration), initial[0])
                )
            result = experiment(scenario, work.parameters, settings, target, mode)
            return result, "Experiment complete. Review forecasts, constraints and metrics."
        except (ValueError, TypeError, OverflowError) as error:
            return no_update, f"Experiment failed: {error}"

    @app.callback(Output("task2-review", "children"), Input("task2-result", "data"))
    def show(result):
        return review(result)

    @app.callback(
        Output("task2-download", "data"),
        Input("task2-export", "n_clicks"),
        State("task2-result", "data"),
        prevent_initial_call=True,
    )
    def export(clicks, result):
        if result is None:
            return no_update
        return dcc.send_string(
            json.dumps(result, allow_nan=False, indent=2), "task2-experiment.json"
        )

    return dict(task2_visibility=visibility, task2_run=run, task2_review=show, task2_export=export)

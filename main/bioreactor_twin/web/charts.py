"""Interactive Plotly views; scientific calculations stay in the services layer."""

import numpy as np
import plotly.graph_objects as go

from bioreactor_twin.hybrid.species import OUTPUTS as NAMES
from bioreactor_twin.web.services import LABELS, UNITS

COLORS = ["#246fe5", "#19a99a", "#df9a36", "#9b6bd3"]


def measured(run, output):
    return (
        run.observed[:, NAMES.index(output)]
        if output in NAMES[:5]
        else run.additional.get(output, np.full(len(run.times), np.nan))
    )


def base_figure(output, message=None):
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white",
        font={"family": "Segoe UI, Arial", "size": 12, "color": "#52637c"},
        margin={"l": 60, "r": 25, "t": 20, "b": 55},
        height=365,
        hovermode="x unified",
        paper_bgcolor="white",
        legend={"orientation": "h", "y": 1.14, "x": 0, "groupclick": "togglegroup"},
        uirevision=output,
        xaxis_title="Elapsed culture time (h)",
        yaxis_title=f"{LABELS[output]} ({UNITS[output]})",
    )
    figure.update_xaxes(gridcolor="#edf1f6", zeroline=False)
    figure.update_yaxes(gridcolor="#edf1f6", zeroline=False, rangemode="tozero")
    if message:
        figure.add_annotation(
            text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )
    return figure


def exploration(runs, selected, output):
    if output == "FEEDS":
        return feed_figure(runs, selected)
    figure = base_figure(output)
    for index, number in enumerate(selected):
        run = runs[int(number)]
        figure.add_trace(
            go.Scatter(
                x=run.times,
                y=measured(run, output),
                name=f"Run {number}",
                legendgroup=f"run-{number}",
                mode="lines+markers",
                line={"color": COLORS[index % 4], "width": 2},
                marker={"size": 6},
                connectgaps=False,
            )
        )
    return figure


def comparison(work, selected, output):
    figure = base_figure(output)
    figure.update_layout(uirevision=f"{output}:" + ",".join(map(str, selected)))
    column = NAMES.index(output)
    for number in selected:
        run = work.runs[int(number)]
        color = COLORS[list(work.runs).index(int(number)) % 4]
        is_cv = work.validation.get("mode") == "cross_validation"
        label = "CV held-out" if is_cv else ("fit" if int(number) in work.trained else "held out")
        if int(number) in work.predictions and column < work.predictions[int(number)].shape[1]:
            sd = work.bands.get(int(number), {}).get(output)
            if sd is not None:
                mean = work.predictions[int(number)][:, column]
                for sign in (1, -1):
                    figure.add_trace(
                        go.Scatter(
                            x=list(range(len(run.events))),
                            y=mean + sign * 1.96 * sd,
                            line={"width": 0},
                            showlegend=False,
                            legendgroup=f"run-{number}",
                            fill="tonexty" if sign == -1 else None,
                            fillcolor="rgba(36,111,229,.10)",
                            hoverinfo="skip",
                        )
                    )
            figure.add_trace(
                go.Scatter(
                    x=list(range(len(run.events))),
                    y=work.predictions[int(number)][:, column],
                    name=f"Run {number} model · {label}",
                    legendgroup=f"run-{number}",
                    line={"color": color},
                )
            )
        figure.add_trace(
            go.Scatter(
                x=run.times[1:],
                y=measured(run, output)[1:],
                name=f"Run {number} assays",
                legendgroup=f"run-{number}",
                mode="markers",
                marker={"color": color, "size": 7, "symbol": "circle-open"},
            )
        )
    return figure


def replicate_figure(info, output, pair):
    figure = base_figure(output)
    column = NAMES.index(output)
    times = info["common_times_h"]
    mean = np.array(info["mean"])[:, column]
    width = 1.96 * info["sd"][column]
    figure.add_trace(
        go.Scatter(
            x=times,
            y=mean + width,
            mode="lines",
            line={"width": 0},
            legendgroup="replicate-spread",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=np.maximum(0, mean - width),
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(36,111,229,.12)",
            name="Descriptive spread (±1.96 SD)",
            legendgroup="replicate-spread",
            legendrank=1100,
            hoverinfo="skip",
        )
    )
    for index, key in enumerate(("first", "second")):
        figure.add_trace(
            go.Scatter(
                x=times,
                y=np.array(info[key])[:, column],
                mode="lines+markers",
                name=f"Run {pair[index]}",
                line={"color": COLORS[index], "width": 2},
            )
        )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=mean,
            mode="lines+markers",
            name="Observed pair mean",
            legendrank=1050,
            line={"color": "#69439b", "width": 2, "dash": "dash"},
            marker={"symbol": "diamond", "size": 6},
            hovertemplate="%{y:.4g}<extra>Observed pair mean</extra>",
        )
    )
    return figure


def simulation_figure(values, baseline, output):
    figure = base_figure(output)
    column = NAMES.index(output)
    if column >= values.shape[1]:
        return base_figure(output, "The active model does not predict this species.")
    for array, name, color, dash in [
        (baseline, "Reference recipe", "#a8b4c5", "dot"),
        (values, "New scenario", "#246fe5", "solid"),
    ]:
        if column >= array.shape[1]:
            continue
        figure.add_trace(
            go.Scatter(
                x=list(range(len(array))),
                y=array[:, column],
                name=name,
                line={"color": color, "width": 2.5, "dash": dash},
            )
        )
    figure.update_xaxes(title="Hours from initialization")
    return figure


def feed_figure(runs, selected):
    figure = go.Figure()
    for index, number in enumerate(selected):
        run = runs[int(number)]
        for column, name in enumerate(("A", "B", "C"), 1):
            hours = np.flatnonzero(run.events[:, column])
            figure.add_trace(
                go.Bar(
                    x=hours,
                    y=run.events[hours, column] * 1000,
                    name=f"Run {number} · Feed {name}",
                    legendgroup=f"run-{number}",
                    marker_color=COLORS[index % 4],
                    opacity=1 - (column - 1) * 0.25,
                    hovertemplate=(
                        "Hour %{x}<br>Feed amount %{y:.4g} mL<extra>%{fullData.name}</extra>"
                    ),
                )
            )
    figure.update_layout(
        template="plotly_white",
        height=400,
        barmode="group",
        xaxis_title="Recorded event hour",
        yaxis_title="Bolus feed volume (mL)",
        legend=dict(groupclick="togglegroup"),
        margin=dict(t=35, b=55),
    )
    return figure

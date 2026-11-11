"""Two-pane laboratory workspace: blue controls and an analytical results canvas."""

from uuid import uuid4

from dash import dash_table, dcc, html

from bioreactor_twin.web.charts import base_figure
from bioreactor_twin.web.equations import equations
from bioreactor_twin.web.services import LABELS, NAMES, RECIPE

TABS = [
    ("explore", "01", "Data & exploration"),
    ("replicates", "02", "Replicate analysis"),
    ("training", "03", "Model training"),
    ("simulation", "04", "Simulation"),
]


def field(label, component, hint=None):
    return html.Div(
        [html.Label(label), component, html.Small(hint) if hint else None], className="field"
    )


def number(identity, value, minimum=0, maximum=None, step=0.1):
    return dcc.Input(
        id=identity,
        type="number",
        value=value,
        min=minimum,
        max=maximum,
        step=step if isinstance(step, int) else "any",
        debounce=True,
        className="number-input",
    )


def action(identity, text, secondary=False):
    return html.Button(
        text,
        id=identity,
        n_clicks=0,
        className="secondary-button" if secondary else "primary-button",
    )


def table(identity, columns, **kwargs):
    return dash_table.DataTable(
        id=identity,
        columns=[{"name": label, "id": key} for key, label in columns],
        data=[],
        page_size=7,
        style_table={"overflowX": "auto"},
        style_cell={
            "color": "#213a58",
            "fontFamily": "Segoe UI, Arial",
            "fontSize": 12,
            "padding": "11px 14px",
            "textAlign": "left",
            "border": "none",
            "borderBottom": "1px solid #edf1f6",
        },
        style_header={
            "backgroundColor": "#f6f8fc",
            "color": "#738099",
            "fontWeight": 600,
            "fontSize": 11,
        },
        **kwargs,
    )


def card(title, body, subtitle=None, class_name=""):
    return html.Section(
        [
            html.Div(
                [html.H3(title), html.P(subtitle) if subtitle else None], className="card-heading"
            ),
            body,
        ],
        className="card " + class_name,
    )


def species_settings():
    grid = table(
        "hybrid-species-settings",
        [("species", "Species"), ("regressor", "Regressor"), ("features", "Manual features")],
        editable=True,
    )
    grid.data = [
        dict(species=name, regressor="inherit", features=features)
        for name, features in (("LAC", "GLC, mu"), ("GLN", "mu"), ("AMM", "mu"))
    ]
    return grid


def task2_controls():
    from bioreactor_twin.control.web import controls

    return controls()


def sidebar():
    return html.Aside(
        [
            html.Div(
                [
                    html.Div("bt", className="brand-mark"),
                    html.Div([html.Strong("Bioreactor lab"), html.Small("DIGITAL TWIN WORKSPACE")]),
                ],
                className="brand",
            ),
            html.Div("PROCESS DEVELOPMENT", className="side-eyebrow"),
            dcc.Tabs(
                id="navigation",
                value="explore",
                vertical=True,
                children=[
                    dcc.Tab(
                        label=f"{num}   {label}",
                        value=value,
                        className="nav-tab",
                        selected_className="nav-tab selected",
                    )
                    for value, num, label in TABS
                ],
                className="navigation",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Your experimental data"),
                            dcc.Upload(
                                id="upload",
                                children=html.Div(
                                    [
                                        html.Span("↑", className="upload-icon"),
                                        html.Strong("Import a workbook"),
                                        html.Small("Drop an .xlsx file or browse"),
                                    ]
                                ),
                                accept=".xlsx",
                                className="upload",
                            ),
                            action("example", "Load supplied dataset", True),
                            field(
                                "Exploration view",
                                dcc.Dropdown(
                                    id="explore-kind",
                                    value="assays",
                                    clearable=False,
                                    options=[
                                        {"label": "Assay concentrations", "value": "assays"},
                                        {"label": "Feed amounts (A, B, C)", "value": "feeds"},
                                    ],
                                    className="light-select",
                                ),
                            ),
                            field(
                                "Runs to explore",
                                dcc.Dropdown(
                                    id="explore-runs",
                                    multi=True,
                                    placeholder="Select runs",
                                    className="light-select",
                                ),
                            ),
                            html.P(
                                (
                                    "Original assays and feed events are preserved. Missing "
                                    "measurements remain missing."
                                ),
                                className="side-note",
                            ),
                        ],
                        id="controls-explore",
                    ),
                    html.Div(
                        [
                            html.H4("Compare repeat experiments"),
                            field(
                                "First replicate",
                                dcc.Dropdown(id="replicate-a", className="light-select"),
                            ),
                            field(
                                "Second replicate",
                                dcc.Dropdown(id="replicate-b", className="light-select"),
                            ),
                            html.P(
                                (
                                    "Choose runs with the same intended operating condition. The "
                                    "nominal pair is Runs 1 and 2."
                                ),
                                className="side-note",
                            ),
                            action("analyze", "Analyze replicates"),
                        ],
                        id="controls-replicates",
                        style={"display": "none"},
                    ),
                    html.Div(
                        [
                            html.H4("Calibrate your digital twin"),
                            field(
                                "Model type",
                                dcc.Dropdown(
                                    id="model-type",
                                    value="first_principles",
                                    clearable=False,
                                    options=[
                                        {
                                            "label": "First-principles · current model",
                                            "value": "first_principles",
                                        },
                                        {
                                            "label": "Hybrid · learned metabolite rates",
                                            "value": "hybrid",
                                        },
                                    ],
                                    className="light-select",
                                ),
                            ),
                            html.Div(id="model-type-note", className="side-note"),
                            html.Div(
                                [
                                    field(
                                        "Learned species",
                                        dcc.Checklist(
                                            id="hybrid-targets",
                                            options=["LAC", "GLN", "AMM"],
                                            value=["LAC", "GLN", "AMM"],
                                        ),
                                    ),
                                    field(
                                        "Rate regressor",
                                        dcc.Dropdown(
                                            id="hybrid-family",
                                            options=[
                                                {"label": x.title(), "value": x}
                                                for x in ("auto", "constant", "ridge", "gp")
                                            ],
                                            value="auto",
                                            clearable=False,
                                        ),
                                    ),
                                    field(
                                        "Feature selection",
                                        dcc.Dropdown(
                                            id="hybrid-feature-mode",
                                            options=[
                                                {"label": x.title(), "value": x}
                                                for x in ("fixed", "manual", "auto")
                                            ],
                                            value="fixed",
                                            clearable=False,
                                        ),
                                    ),
                                    field(
                                        "Manual features",
                                        dcc.Dropdown(
                                            id="hybrid-features",
                                            options=[
                                                "VCD",
                                                "viability",
                                                "GLC",
                                                "GLU",
                                                "mu",
                                                "time",
                                            ],
                                            value=["GLC", "mu"],
                                            multi=True,
                                        ),
                                    ),
                                    species_settings(),
                                    html.Small("Exploratory selection: only three conditions."),
                                ],
                                id="hybrid-controls",
                                style={"display": "none"},
                            ),
                            field(
                                "Training method",
                                dcc.Dropdown(
                                    id="training-mode",
                                    value="train_test",
                                    clearable=False,
                                    options=[
                                        {"label": "Train / test split", "value": "train_test"},
                                        {
                                            "label": "Grouped cross-validation",
                                            "value": "cross_validation",
                                        },
                                    ],
                                    className="light-select",
                                ),
                            ),
                            html.Div(
                                "Replicate groups receive one combined condition weight.",
                                className="side-note",
                            ),
                            html.Div(
                                [
                                    field("Multi-starts", number("starts", 4, 1, 16, 1)),
                                    field(
                                        "Evaluations / start",
                                        number("evaluations", 250, 10, 1000, 10),
                                    ),
                                ],
                                className="field-grid",
                            ),
                            html.P(
                                (
                                    "Multi-substrate Monod · shared kinetics · "
                                    "exact feed and sample "
                                    "balances"
                                ),
                                className="side-note",
                            ),
                            action("train", "Train model"),
                            action("load-hybrid", "Load calibrated hybrid", True),
                            html.Div(
                                id="saved-hybrid-status", className="side-note", role="status"
                            ),
                            html.Div(id="training-request", className="side-note", role="status"),
                            html.Div(id="training-status", className="side-note", role="status"),
                            action("download-model", "Export model parameters", True),
                        ],
                        id="controls-training",
                        style={"display": "none"},
                    ),
                    html.Div(
                        [
                            html.H4("Explore a new operating recipe"),
                            task2_controls(),
                            field(
                                "Reference run",
                                dcc.Dropdown(id="reference-run", className="light-select"),
                            ),
                            action("load-recipe", "Load reference conditions", True),
                            html.Div(
                                [
                                    field("Duration (h)", number("duration", 258, 1, 500, 1)),
                                    field(
                                        "Initial volume (mL)", number("volume", 199, 1, 10000, 1)
                                    ),
                                ],
                                className="field-grid",
                            ),
                            html.Div(
                                [
                                    field("Feed A scale", number("scale-a", 1, 0, 5, 0.1)),
                                    field("Feed B scale", number("scale-b", 1, 0, 5, 0.1)),
                                ],
                                className="field-grid",
                            ),
                            field("Feed B strength vs target", number("strength", 1, 0.01, 5, 0.1)),
                            html.Details(
                                [
                                    html.Summary("Initial cell and metabolite levels"),
                                    field("Initial culture age (h)", number("initial-age", 0)),
                                    field("Glutamine (g/L)", number("initial-GLN", 0.03)),
                                    field("Ammonia (reported g/L)", number("initial-AMM", 0.01)),
                                    *[
                                        field(
                                            name
                                            + " ("
                                            + ("million/mL" if name in ("VCD", "TCD") else "g/L")
                                            + ")",
                                            number(f"initial-{name}", value, 0, None, 0.01),
                                        )
                                        for name, value in zip(
                                            NAMES, [0.3, 0.35, 6, 0.5, 0.2], strict=True
                                        )
                                    ],
                                ],
                                className="initial-details",
                            ),
                            action("simulate", "Run simulation"),
                            html.Div(id="simulation-status", className="side-note", role="status"),
                            action("download-simulation", "Export simulation CSV", True),
                        ],
                        id="controls-simulation",
                        style={"display": "none"},
                    ),
                ],
                className="side-controls",
            ),
            html.Div(
                [
                    html.Span(className="status-dot"),
                    "Python engine",
                    html.Span("1 h", className="step-badge"),
                ],
                className="side-footer",
            ),
        ],
        className="sidebar",
    )


def metric_strip(identity):
    return html.Div(id=identity, className="metric-strip")


def results():
    return html.Main(
        [
            html.Header(
                [
                    html.Div(
                        [
                            html.Span("WORKSPACE", className="eyebrow"),
                            html.Span(" / CHO fed-batch", className="breadcrumb"),
                        ]
                    ),
                    html.Div(
                        [html.Span(className="status-dot"), html.Span("Local workspace")],
                        className="local-badge",
                    ),
                ],
                className="topbar",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "EXPERIMENTAL INSIGHTS", id="page-eyebrow", className="eyebrow"
                            ),
                            html.H1("Data exploration", id="page-title"),
                            html.P(
                                (
                                    "Explore your batches, understand their trajectories, "
                                    "and build a "
                                    "reliable model."
                                ),
                                id="page-subtitle",
                            ),
                        ]
                    ),
                    html.Button(
                        "↓  Export observations",
                        id="download-data",
                        n_clicks=0,
                        className="export-button",
                    ),
                ],
                className="page-heading",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("▤", className="file-icon"),
                            html.Div(
                                [html.Strong(id="dataset-name"), html.Small(id="dataset-detail")]
                            ),
                        ],
                        className="dataset-file",
                    ),
                    html.Span("GLC · GLU · LAC · VCD · TCD", className="channel-badge"),
                ],
                className="dataset-banner",
            ),
            html.Div(id="message", role="status", className="message"),
            html.Div(
                [
                    html.Label("Display"),
                    dcc.Dropdown(
                        id="output",
                        value="VCD",
                        clearable=False,
                        options=[
                            {"label": LABELS[n] if n in ("GLC", "GLU", "LAC") else n, "value": n}
                            for n in (*NAMES, "GLN", "AMM")
                        ],
                        className="output-dropdown",
                    ),
                ],
                className="output-selector",
            ),
            html.Div(
                [
                    metric_strip("explore-metrics"),
                    card(
                        "Culture trajectories",
                        dcc.Graph(
                            id="explore-chart",
                            figure=base_figure("VCD"),
                            config={"displaylogo": False},
                        ),
                        "Daily observations at their actual sampling times",
                    ),
                    html.Div(
                        [
                            card(
                                "Assay table",
                                table(
                                    "assay-table",
                                    [
                                        ("run", "Run"),
                                        ("time_h", "Time (h)"),
                                        *[(n, n) for n in (*NAMES, "GLN", "AMM")],
                                    ],
                                    sort_action="native",
                                ),
                                "Original observations in reported units",
                            ),
                            card(
                                "Batch overview",
                                html.Div(
                                    [
                                        table(
                                            "run-table",
                                            [
                                                ("run", "Run"),
                                                ("assays", "Assays"),
                                                ("duration_h", "Hours"),
                                                ("feed_B_mL", "Feed B (mL)"),
                                                ("B_strength", "B strength"),
                                                ("replicate_group", "Replicate group"),
                                            ],
                                            row_selectable="multi",
                                            selected_rows=[],
                                        ),
                                        action("mark-replicates", "Mark selected as replicates"),
                                        action("ungroup-replicates", "Make selected independent"),
                                        html.P(id="replicate-group-status", role="status"),
                                    ]
                                ),
                                "Select rows to mark replicates or make them independent.",
                            ),
                        ],
                        className="two-cards",
                    ),
                ],
                id="results-explore",
            ),
            html.Div(
                [
                    metric_strip("replicate-metrics"),
                    card(
                        "Replicate trajectories & spread",
                        dcc.Graph(
                            id="replicate-chart",
                            figure=base_figure("VCD"),
                            config={"displaylogo": False},
                        ),
                        "Exact common sampling times; no interpolated assay labels",
                    ),
                    card(
                        "Repeatability by output",
                        table(
                            "replicate-table",
                            [("output", "Output"), ("sd", "Descriptive SD"), ("units", "Unit")],
                        ),
                        (
                            "Two runs describe repeatability; they cannot establish guaranteed "
                            "prediction coverage."
                        ),
                    ),
                    html.P(id="replicate-note", className="result-note"),
                ],
                id="results-replicates",
                style={"display": "none"},
            ),
            html.Div(
                [
                    metric_strip("training-metrics"),
                    card(
                        "Hybrid model review",
                        html.Div(
                            [
                                dcc.Tabs(
                                    id="hybrid-view",
                                    value="diagnostics",
                                    children=[
                                        dcc.Tab(
                                            label="Regression diagnostics", value="diagnostics"
                                        ),
                                        dcc.Tab(label="Learned rates", value="rates"),
                                        dcc.Tab(label="Features & model", value="features"),
                                    ],
                                ),
                                dcc.Checklist(
                                    id="diagnostic-training",
                                    options=[
                                        {
                                            "label": "Include training points (train/test only)",
                                            "value": "include",
                                        }
                                    ],
                                    value=[],
                                ),
                                html.Div(id="hybrid-review"),
                            ]
                        ),
                        "Trajectories remain in the model/experiment graph below.",
                    ),
                    card(
                        "Validation plan",
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.P(
                                            "Mark the experiments to reserve for testing. "
                                            "Unticked experiments are used for training.",
                                            className="result-note",
                                        ),
                                        table(
                                            "split-table",
                                            [
                                                ("run", "Experiment"),
                                                ("group", "Replicate condition"),
                                                ("assays", "Assays"),
                                                ("strength", "Feed B strength"),
                                            ],
                                            row_selectable="multi",
                                            selected_rows=[],
                                        ),
                                    ],
                                    id="test-plan",
                                ),
                                html.Div(
                                    [
                                        table(
                                            "fold-table",
                                            [
                                                ("fold", "Fold"),
                                                ("training", "Training experiments"),
                                                ("test", "Held-out experiments"),
                                            ],
                                        )
                                    ],
                                    id="cv-plan",
                                    style={"display": "none"},
                                ),
                                html.P(id="split-note", className="result-note"),
                            ]
                        ),
                        "Whole-experiment splits; replicate runs always stay together.",
                    ),
                    card(
                        "Model vs. experiment",
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Visible runs"),
                                        dcc.Dropdown(
                                            id="plot-runs",
                                            multi=True,
                                            placeholder="Select runs to visualize",
                                            className="plot-run-dropdown",
                                        ),
                                        html.Small(
                                            "Legend clicks toggle model and assays together."
                                        ),
                                    ],
                                    className="plot-toolbar",
                                ),
                                dcc.Graph(
                                    id="training-chart",
                                    figure=base_figure("VCD"),
                                    config={"displaylogo": False},
                                ),
                            ]
                        ),
                        "Predictions start from the first assay, with no later state resets",
                    ),
                    html.Div(
                        [
                            card(
                                "Model equations and parameter locations",
                                html.Div(equations(False), id="model-equations"),
                            ),
                            card(
                                "Fitted parameters",
                                table(
                                    "parameter-table",
                                    [("parameter", "Parameter"), ("value", "Value")],
                                ),
                            ),
                            card(
                                "Prediction error · RMSE",
                                table(
                                    "score-table",
                                    [
                                        ("run", "Run"),
                                        ("split", "Use"),
                                        *[(n, n) for n in (*NAMES, "GLN", "AMM")],
                                    ],
                                ),
                                "Cells: million/mL. Metabolites: g/L.",
                            ),
                        ],
                        className="two-cards",
                    ),
                    html.P(id="training-note", className="result-note"),
                ],
                id="results-training",
                style={"display": "none"},
            ),
            html.Div(
                [
                    html.Div(
                        [card("Task 2 experiment", html.Div(id="task2-review"))],
                        id="task2-results",
                        style={"display": "none"},
                    ),
                    metric_strip("simulation-metrics"),
                    card(
                        "Operating scenario",
                        dcc.Graph(
                            id="simulation-chart",
                            figure=base_figure(
                                "VCD", "Load a reference recipe, then run a simulation."
                            ),
                            config={"displaylogo": False},
                        ),
                        "Compare changed operating conditions with the reference recipe",
                    ),
                    card(
                        "Feed & sampling recipe",
                        html.Div(
                            [
                                html.P(
                                    (
                                        "Edit amounts and times below. "
                                        "Scales apply to A / B volumes; "
                                        "strength changes B concentration only."
                                    ),
                                    className="result-note",
                                ),
                                table(
                                    "recipe-table",
                                    [
                                        ("hour", "Hour"),
                                        ("sample_mL", "Sample (mL)"),
                                        ("feed_A_mL", "Feed A (mL)"),
                                        ("feed_B_mL", "Feed B (mL)"),
                                        ("feed_C_mL", "Feed C (mL)"),
                                    ],
                                    editable=True,
                                    row_deletable=True,
                                    style_data_conditional=[
                                        {"if": {"column_id": key}, "backgroundColor": "#f5f9ff"}
                                        for key in RECIPE
                                    ],
                                ),
                                html.Button(
                                    "+ Add recipe event",
                                    id="add-event",
                                    n_clicks=0,
                                    className="table-button",
                                ),
                            ]
                        ),
                        "Times are hours from the initial conditions; samples precede feeds.",
                    ),
                    html.P(
                        (
                            "Scenario predictions are conditional on this baseline model. "
                            "Changes outside the observed recipe range need experimental "
                            "validation."
                        ),
                        className="result-note",
                    ),
                ],
                id="results-simulation",
                style={"display": "none"},
            ),
            html.Footer(
                [html.Span("Bioreactor lab"), html.Span(id="model-label")], className="main-footer"
            ),
        ],
        className="results",
    )


def work_indicator(message, identity="request-busy", hidden=False):
    return html.Div(
        [html.Img(src="/assets/bioreactor-loader.svg", alt=""), html.Span(message)],
        id=identity,
        hidden=hidden,
        className="work-indicator",
        role="status",
    )


def layout():
    workspace = html.Div(
        [
            dcc.Store(id="session", data=str(uuid4()), storage_type="session"),
            dcc.Store(id="dataset-version", data=0),
            dcc.Store(id="model-version", data=0),
            dcc.Store(id="split-state"),
            dcc.Store(id="replicate-result"),
            dcc.Store(id="simulation-result"),
            dcc.Interval(id="job-poll", interval=1500),
            dcc.Download(id="download"),
            sidebar(),
            results(),
            work_indicator("Training model...", "training-busy", hidden=True),
        ],
        className="app-shell",
    )

    return dcc.Loading(
        workspace,
        custom_spinner=work_indicator("Working..."),
        overlay_style={"visibility": "visible"},
        show_initially=False,
        delay_show=250,
        target_components={
            "dataset-version": "data",
            "explore-chart": "figure",
            "replicate-chart": "figure",
            "training-request": "children",
            "saved-hybrid-status": "children",
            "simulation-result": "data",
            "task2-result": "data",
            "download": "data",
            "task2-download": "data",
        },
    )

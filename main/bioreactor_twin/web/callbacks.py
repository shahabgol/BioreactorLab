"""Dash adapters with thin, directly testable callbacks over the laboratory services."""

from dataclasses import asdict

import numpy as np
import pandas as pd
from dash import Input, Output, State, ctx, dcc, html, no_update

from bioreactor_twin.calibration.validation import linked_test_rows, validation_splits
from bioreactor_twin.data.dataset import NAMES, condition_key
from bioreactor_twin.hybrid.species import OUTPUTS
from bioreactor_twin.simulation.nitrogen import simulate_mechanistic
from bioreactor_twin.web.charts import (
    base_figure,
    comparison,
    exploration,
    replicate_figure,
    simulation_figure,
)
from bioreactor_twin.web.layout import TABS
from bioreactor_twin.web.services import (
    RECIPE,
    UNITS,
    observations,
    parameter_rows,
    recipe_from_run,
    replicate_analysis,
    run_summary,
    scenario_defaults,
    scenario_run,
)

PAGES = {
    "explore": (
        "EXPERIMENTAL INSIGHTS",
        "Data exploration",
        "Explore your batches, understand their trajectories, and build a reliable model.",
    ),
    "replicates": (
        "PROCESS REPEATABILITY",
        "Replicate analysis",
        "Quantify run-to-run variation before interpreting model accuracy.",
    ),
    "training": (
        "MODEL DEVELOPMENT",
        "Model training",
        "Calibrate shared kinetics and compare predictions with the experimental assays.",
    ),
    "simulation": (
        "VIRTUAL EXPERIMENTS",
        "Operating simulations",
        "Change initial conditions and feed recipes, then explore the predicted response.",
    ),
}


def metric(name, value, detail):
    return html.Div(
        [html.Div(name, className="metric-name"), html.Strong(str(value)), html.Small(detail)],
        className="metric",
    )


def register_callbacks(app, lab):
    functions = {}

    @app.callback(
        [Output(f"controls-{key}", "style") for key, _, _ in TABS]
        + [Output(f"results-{key}", "style") for key, _, _ in TABS]
        + [
            Output("page-eyebrow", "children"),
            Output("page-title", "children"),
            Output("page-subtitle", "children"),
        ],
        Input("navigation", "value"),
    )
    def navigate(tab):
        styles = [{} if key == tab else {"display": "none"} for key, _, _ in TABS]
        return [*styles, *styles, *PAGES[tab]]

    functions["navigate"] = navigate

    @app.callback(
        Output("dataset-version", "data"),
        Output("message", "children"),
        Input("upload", "contents"),
        Input("example", "n_clicks"),
        State("upload", "filename"),
        State("session", "data"),
        State("dataset-version", "data"),
        prevent_initial_call=True,
    )
    def import_data(contents, clicks, filename, session, revision):
        try:
            work = lab.import_data(
                session, None if ctx.triggered_id == "example" else contents, filename
            )
            return (
                revision + 1,
                f"Loaded {len(work.runs)} runs. Source data retained; model state refreshed.",
            )
        except Exception as error:
            return no_update, f"Import could not be completed: {error}"

    functions["import"] = import_data

    @app.callback(
        [
            Output(identity, prop)
            for identity in (
                "explore-runs",
                "plot-runs",
                "replicate-a",
                "replicate-b",
                "reference-run",
            )
            for prop in ("options", "value")
        ]
        + [
            Output("dataset-name", "children"),
            Output("dataset-detail", "children"),
            Output("split-table", "data"),
        ],
        Input("dataset-version", "data"),
        Input("session", "data"),
    )
    def populate(revision, session):
        work = lab.get(session)
        numbers = list(work.runs)
        options = [{"label": f"Run {n}", "value": n} for n in numbers]
        first = numbers[0] if numbers else None
        second = numbers[1] if len(numbers) > 1 else first
        count = sum(len(r.times) for r in work.runs.values())
        detail = f"{len(numbers)} runs · {count} retained assays · hourly process ledger"
        return [
            options,
            numbers,
            options,
            numbers,
            options,
            first,
            options,
            second,
            options,
            first,
            work.filename,
            detail,
            [
                {
                    "run": n,
                    "group": condition_key(work.runs[n]),
                    "assays": len(work.runs[n].times),
                    "strength": work.runs[n].strength,
                }
                for n in numbers
            ],
        ]

    functions["populate"] = populate

    @app.callback(
        Output("dataset-version", "data", allow_duplicate=True),
        Output("replicate-group-status", "children"),
        Input("mark-replicates", "n_clicks"),
        Input("ungroup-replicates", "n_clicks"),
        State("run-table", "selected_rows"),
        State("run-table", "data"),
        State("session", "data"),
        State("dataset-version", "data"),
        prevent_initial_call=True,
    )
    def set_replicates(mark, clear, selected, rows, session, revision):
        try:
            numbers = [rows[i]["run"] for i in selected or []]
            message = lab.set_replicates(session, numbers, ctx.triggered_id == "ungroup-replicates")
            return revision + 1, message
        except (ValueError, IndexError) as error:
            return no_update, str(error)

    functions["set_replicates"] = set_replicates

    @app.callback(Output("model-equations", "children"), Input("model-type", "value"))
    def model_equations(kind):
        from bioreactor_twin.web.equations import equations

        return equations(kind == "hybrid")

    functions["equations"] = model_equations

    @app.callback(
        Output("split-table", "selected_rows"),
        Output("split-state", "data"),
        Output("split-note", "children"),
        Output("fold-table", "data"),
        Output("test-plan", "style"),
        Output("cv-plan", "style"),
        Input("training-mode", "value"),
        Input("split-table", "selected_rows"),
        Input("dataset-version", "data"),
        State("session", "data"),
        State("split-state", "data"),
    )
    def validation_plan(mode, selected, revision, session, previous):
        work = lab.get(session)
        numbers = list(work.runs)
        if previous is None or previous["revision"] != revision:
            selected = [i for i, n in enumerate(numbers) if n in (1, 2)] if len(numbers) > 2 else []
            prior = []
        else:
            prior = previous["selected"]
        selected = linked_test_rows(numbers, selected, prior, work.runs)
        tests = [numbers[i] for i in selected]
        state = {
            "revision": revision,
            "selected": selected,
            "test_runs": tests,
            "mode": mode,
            "valid": True,
        }
        try:
            splits = validation_splits(work.runs, mode, tests)
            rows = [
                {
                    "fold": i + 1,
                    "training": ", ".join(map(str, split.training)),
                    "test": ", ".join(map(str, split.test)) or "None",
                }
                for i, split in enumerate(splits)
            ]
            note = (
                "Each replicate group is held out once, always in the same fold. "
                "Charts and RMSE use out-of-fold predictions. "
                "A separate all-data refit "
                "is used for simulation."
                if mode == "cross_validation"
                else "Check test rows below. Selecting a replicate selects its whole group; "
                "unchecking a member removes that group. "
                "No checked rows means an all-data refit without independent test performance."
            )
        except ValueError as error:
            state["valid"] = False
            note, rows = str(error), []
        return (
            selected,
            state,
            note,
            rows,
            ({"display": "none"} if mode == "cross_validation" else {}),
            ({} if mode == "cross_validation" else {"display": "none"}),
        )

    functions["validation_plan"] = validation_plan

    @app.callback(Output("model-type-note", "children"), Input("model-type", "value"))
    def model_type_note(model_type):
        if model_type == "hybrid":
            return (
                "Mechanistic VCD/TCD and glucose/glutamate; learned signed rates for "
                "lactate, glutamine and ammonia. No metabolite feedback into growth."
            )
        return (
            "Monod growth and mechanistic balances for glucose, glutamate, lactate, "
            "glutamine and ammonia."
        )

    functions["model_type_note"] = model_type_note

    @app.callback(
        Output("explore-chart", "figure"),
        Output("assay-table", "data"),
        Output("run-table", "data"),
        Output("explore-metrics", "children"),
        Input("explore-runs", "value"),
        Input("output", "value"),
        Input("dataset-version", "data"),
        Input("explore-kind", "value"),
        State("session", "data"),
    )
    def explore(selected, output, revision, kind, session):
        runs = lab.get(session).runs
        chosen = [n for n in (selected or []) if int(n) in runs]
        frame = observations(runs, chosen)
        hours = max((len(runs[int(n)].events) - 1 for n in chosen), default=0)
        cards = [
            metric("Selected batches", len(chosen), "Independent culture trajectories"),
            metric("Assay panels", len(frame), "Daily measurements; actual timestamps"),
            metric("Longest culture", f"{hours} h", "Feed and sampling events retained"),
            metric("Model channels", "5", "2 cell counts · 3 metabolites"),
        ]
        return (
            exploration(runs, chosen, "FEEDS" if kind == "feeds" else output),
            frame.round(5).to_dict("records"),
            [
                {**row, "replicate_group": condition_key(runs[row["run"]])}
                for row in run_summary(runs)
            ],
            cards,
        )

    functions["explore"] = explore

    @app.callback(
        Output("replicate-chart", "figure"),
        Output("replicate-table", "data"),
        Output("replicate-metrics", "children"),
        Output("replicate-note", "children"),
        Input("analyze", "n_clicks"),
        Input("output", "value"),
        Input("model-version", "data"),
        Input("replicate-a", "value"),
        Input("replicate-b", "value"),
        Input("dataset-version", "data"),
        State("session", "data"),
    )
    def replicates(clicks, output, model_revision, first, second, revision, session):
        try:
            work = lab.get(session)
            info = replicate_analysis(work.runs, [first, second], work.parameters, work.nitrogen)
            rows = [
                {"output": n, "sd": round(info["sd"][i], 5), "units": UNITS[n]}
                for i, n in enumerate(OUTPUTS[: len(info["sd"])])
            ]
            adjusted = output in info["adjusted_outputs"]
            cards = [
                metric(
                    "Replicate pair", f"{first} / {second}", "Confirm identical intended conditions"
                ),
                metric(
                    "Matched assays",
                    len(info["common_times_h"]),
                    "Exact common times after initialization",
                ),
                metric(
                    f"{output} spread", f"{info['sd'][OUTPUTS.index(output)]:.3g}", UNITS[output]
                ),
                metric(
                    "Adjustment",
                    "Model" if adjusted else "Raw",
                    "Feed / initialization effects" if adjusted else "Observed pair differences",
                ),
            ]
            note = (
                "Descriptive repeatability from two batches includes assay and "
                "process variation. The dashed line is the observed pair mean. "
                "The shaded spread is descriptive. "
                "The adjustment badge applies to the displayed species."
            )
            return replicate_figure(info, output, [first, second]), rows, cards, note
        except (ValueError, KeyError, TypeError) as error:
            return base_figure(output, "Choose a valid replicate pair."), [], [], str(error)

    functions["replicates"] = replicates

    @app.callback(
        Output("training-request", "children"),
        Input("train", "n_clicks"),
        State("starts", "value"),
        State("evaluations", "value"),
        State("training-mode", "value"),
        State("model-type", "value"),
        State("split-state", "data"),
        State("session", "data"),
        State("hybrid-targets", "value"),
        State("hybrid-family", "value"),
        State("hybrid-feature-mode", "value"),
        State("hybrid-features", "value"),
        State("hybrid-species-settings", "data"),
        prevent_initial_call=True,
    )
    def train(
        clicks,
        starts,
        evaluations,
        mode,
        model_type,
        plan,
        session,
        targets=None,
        family="auto",
        feature_mode="fixed",
        features=None,
        species_settings=None,
    ):
        try:
            if not plan or not plan["valid"] or plan["mode"] != mode:
                raise ValueError("Review the validation plan before training.")
            return lab.train(
                session,
                list(lab.get(session).runs),
                starts,
                evaluations,
                mode=mode,
                test_runs=plan["test_runs"],
                model_type=model_type,
                hybrid_options=dict(
                    targets=("LAC", "GLN", "AMM") if targets is None else targets,
                    family=family,
                    feature_mode=feature_mode,
                    features=features or [],
                    species_settings=species_settings,
                ),
            )
        except (ValueError, TypeError) as error:
            return str(error)

    functions["train"] = train

    @app.callback(
        Output("model-version", "data"),
        Output("training-status", "children"),
        Output("train", "disabled"),
        Output("download-model", "disabled"),
        Output("training-busy", "hidden"),
        Input("job-poll", "n_intervals"),
        Input("dataset-version", "data"),
        Input("model-type", "value"),
        Input("split-state", "data"),
        State("session", "data"),
        State("model-version", "data"),
    )
    def poll(interval, revision, model_type, plan, session, model_revision):
        work = lab.get(session)
        job = work.job
        status = job["status"]
        text = work.label
        if status in ("queued", "running"):
            text = (
                f"{status.title()} · {job.get('stage', 'Queued')} · "
                f"{job['completed']} / {job['starts']} starts complete"
            )
        elif status == "failed":
            text = "Training failed: " + job["error"]
        elif status == "done":
            text = (
                "Training complete. Model and results have been updated."
                if job.get("converged", True)
                else "Evaluation budget reached. Candidate results are available; "
                "review before use."
            )
        return (
            work.version if work.version != model_revision else no_update,
            text,
            status in ("queued", "running")
            or model_type not in ("first_principles", "hybrid")
            or not (plan and plan["valid"]),
            work.parameters is None,
            status not in ("queued", "running"),
        )

    functions["poll"] = poll

    @app.callback(
        Output("training-chart", "figure"),
        Output("parameter-table", "data"),
        Output("score-table", "data"),
        Output("training-metrics", "children"),
        Output("training-note", "children"),
        Output("model-label", "children"),
        Input("model-version", "data"),
        Input("dataset-version", "data"),
        Input("output", "value"),
        Input("plot-runs", "value"),
        State("session", "data"),
    )
    def training_results(model_revision, revision, output, selected, session):
        work = lab.get(session)
        is_cv = work.validation.get("mode") == "cross_validation"
        held = [n for n in work.runs if n not in work.trained]
        cards = [
            metric(
                "Active model refit" if is_cv else "Training runs",
                ", ".join(map(str, work.trained)) or "—",
                "Shared kinetic parameters",
            ),
            metric(
                "Validation folds" if is_cv else "Held-out runs",
                str(len(work.validation["folds"]))
                if is_cv
                else (", ".join(map(str, held)) or "None"),
                "Grouped out-of-fold predictions"
                if is_cv
                else ("Unseen by this fit" if held else "All-run refit; no held-out performance"),
            ),
            metric("Integration step", "1 h", "RK4 with exact bolus events"),
            metric(
                "Model structure",
                "Hybrid" if work.hybrid else "Monod",
                "Mechanistic core + learned rates"
                if work.hybrid
                else (
                    "Includes GLN and AMM" if work.nitrogen else "GLC + GLU; bidirectional lactate"
                ),
            ),
        ]
        scores = [
            {
                **{k: round(v, 4) if k != "run" and v is not None else v for k, v in row.items()},
                "split": "CV held-out"
                if is_cv
                else ("Training" if row["run"] in work.trained else "Held out"),
            }
            for row in work.scores
        ]
        note = (
            "First assays initialize each run and are not scored. Runs 1/2 "
            "receive one combined condition weight. Fitting all runs reports "
            "calibration error, not independent validation. Glucose saturation "
            "and uptake parameters can be weakly identified; review "
            "biochemical residuals."
        )
        if is_cv:
            note = (
                "Every plotted model trajectory and RMSE is out-of-fold: its replicate condition "
                "was excluded from that fold's fit. Displayed parameters are the separate "
                "all-data refit used by simulation. Initial assays are not scored."
            )
        chosen = [n for n in (selected or []) if int(n) in work.runs]
        return (
            comparison(work, chosen, output),
            parameter_rows(work.parameters, work.nitrogen) if work.parameters else [],
            scores,
            cards,
            note
            + (
                " GP shading: conditional ±1.96 posterior SD under working weights; "
                "not calibrated process coverage. Invalid trajectories: "
                + str({n: v for n, v in work.invalid.items() if v})
                + " Consumption limited at (run, species, hour): "
                + str(work.validation.get("depletion_hours", {}))
                if work.hybrid
                else ""
            ),
            work.label,
        )

    functions["training_results"] = training_results

    @app.callback(
        [Output("recipe-table", "data")]
        + [
            Output(identity, "value")
            for identity in ("duration", "volume", "strength", *[f"initial-{n}" for n in NAMES])
        ],
        Input("load-recipe", "n_clicks"),
        Input("add-event", "n_clicks"),
        Input("reference-run", "value"),
        State("recipe-table", "data"),
        State("session", "data"),
    )
    def recipe(load_clicks, add_clicks, reference, rows, session):
        if ctx.triggered_id == "add-event":
            return [rows + [dict.fromkeys(RECIPE, 0)], *[no_update] * 8]
        work = lab.get(session)
        if reference not in work.runs:
            return [[], *[no_update] * 8]
        run = work.runs[reference]
        defaults = scenario_defaults(run)
        return [
            recipe_from_run(run),
            *[defaults[n] for n in ("duration", "volume", "strength", *NAMES)],
        ]

    functions["recipe"] = recipe

    @app.callback(
        Output("simulation-result", "data"),
        Output("simulation-status", "children"),
        Input("simulate", "n_clicks"),
        Input("model-version", "data"),
        Input("dataset-version", "data"),
        State("reference-run", "value"),
        State("recipe-table", "data"),
        *[
            State(identity, "value")
            for identity in (
                "duration",
                "volume",
                "strength",
                "scale-a",
                "scale-b",
                *[f"initial-{n}" for n in NAMES],
            )
        ],
        State("session", "data"),
        State("initial-GLN", "value"),
        State("initial-AMM", "value"),
        State("initial-age", "value"),
        prevent_initial_call=True,
    )
    def simulation(
        clicks,
        model_revision,
        revision,
        reference,
        rows,
        duration,
        volume,
        strength,
        scale_a,
        scale_b,
        vcd,
        tcd,
        glc,
        glu,
        lac,
        session,
        gln=None,
        amm=None,
        culture_age=0,
    ):
        if ctx.triggered_id != "simulate":
            return None, "Run the simulation to evaluate the current model and recipe."
        try:
            work = lab.get(session)
            if work.parameters is None:
                raise ValueError("Train a model before simulating a new recipe.")
            settings = dict(
                duration=duration,
                culture_age=culture_age,
                volume=volume,
                strength=strength,
                scale_A=scale_a,
                scale_B=scale_b,
                **dict(zip(NAMES, [vcd, tcd, glc, glu, lac], strict=True)),
            )
            if work.hybrid or work.nitrogen:
                settings.update(GLN=gln, AMM=amm)
            run = scenario_run(settings, rows)
            depletion = {}
            source = work.runs[reference]
            baseline_run = scenario_run(scenario_defaults(source), recipe_from_run(source))
            if work.hybrid:
                values, _, invalid = work.hybrid.simulate(run, diagnostics=depletion)
                if invalid:
                    raise ValueError(str(invalid))
                baseline, _, _ = work.hybrid.simulate(baseline_run)
            else:
                values = simulate_mechanistic(run, work.parameters, work.nitrogen)
                baseline = simulate_mechanistic(baseline_run, work.parameters, work.nitrogen)
            result = {
                "values": values.tolist(),
                "baseline": baseline.tolist(),
                "model": work.label,
                "volume": (run.volumes * 1000).tolist(),
                "parameters": asdict(work.parameters),
                "depletion_hours": depletion,
                "columns": list(OUTPUTS[: values.shape[1]]),
            }
            message = "Simulation complete. Explore outputs or export the hourly results."
            if depletion:
                message += f" Consumption limited at (species, hour): {depletion}."
            return result, message
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            return no_update, f"Simulation could not be completed: {error}"

    functions["simulation"] = simulation

    @app.callback(
        Output("simulation-chart", "figure"),
        Output("simulation-metrics", "children"),
        Output("download-simulation", "disabled"),
        Input("simulation-result", "data"),
        Input("output", "value"),
    )
    def simulation_results(result, output):
        if result is None:
            return base_figure(output, "Load a reference recipe, then run a simulation."), [], True
        values = np.array(result["values"])
        cards = [
            metric("Peak VCD", f"{max(values[:, 0]):.2f}", "million cells/mL"),
            metric("Final TCD", f"{values[-1, 1]:.2f}", "million cells/mL"),
            metric("Final glucose", f"{values[-1, 2]:.2f}", "g/L"),
            metric("Final volume", f"{result['volume'][-1]:.1f}", "mL after final events"),
        ]
        return simulation_figure(values, np.array(result["baseline"]), output), cards, False

    functions["simulation_results"] = simulation_results

    @app.callback(
        Output("download", "data"),
        Input("download-data", "n_clicks"),
        Input("download-model", "n_clicks"),
        Input("download-simulation", "n_clicks"),
        State("session", "data"),
        State("simulation-result", "data"),
        prevent_initial_call=True,
    )
    def download(data_clicks, model_clicks, simulation_clicks, session, result):
        if ctx.triggered_id == "download-model":
            return dcc.send_string(lab.model_json(session), "bioreactor_model.json")
        if ctx.triggered_id == "download-simulation" and result is not None:
            frame = pd.DataFrame(result["values"], columns=result.get("columns", NAMES))
            frame.insert(0, "time_h", range(len(frame)))
            frame["volume_mL_after_events"] = result["volume"]
            return dcc.send_data_frame(frame.to_csv, "simulation.csv", index=False)
        return dcc.send_data_frame(
            observations(lab.get(session).runs).to_csv, "observations.csv", index=False
        )

    functions["download"] = download
    from bioreactor_twin.hybrid.web import register_hybrid

    functions.update(register_hybrid(app, lab))
    from bioreactor_twin.control.web import register

    functions.update(register(app, lab))
    return functions

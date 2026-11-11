"""Synthetic plant trials and causal forecast scoring, independent of Dash."""

from dataclasses import asdict, replace

import numpy as np

from bioreactor_twin.calibration.fit import FLOORS
from bioreactor_twin.control.dynamics import advance, forecast, reachability
from bioreactor_twin.control.mpc import choose_dose
from bioreactor_twin.control.observer import Observer
from bioreactor_twin.simulation.engine import event, initialize, observe


def trial(run, p, settings, target=None, controlled=False, adaptive=False, disturbed=True):
    settings.validate()
    if run.times[0] != 0:
        raise ValueError("Use a scenario recipe starting at zero.")
    steps = len(run.events) - 1
    state = initialize(run)
    observers = {
        name: Observer(state.copy(), p, settings, name == "Adaptive")
        for name in ("Frozen", "State correction", "Adaptive")
    }
    rng = np.random.default_rng(settings.seed)
    delivered = {0: observe(state)}
    pending = []
    truth = []
    estimates = {name: [] for name in observers}
    factors = []
    maintenance_factors = []
    actions = []
    diagnostics = []
    checks = []
    violations = []
    samples = []
    actual_events = run.events.copy()
    issued = {name: [] for name in (*observers, "Oracle")}
    for hour in range(steps + 1):
        actual_factor = settings.multiplier if disturbed and hour >= settings.disturbance else 1.0
        truth.append(observe(state))
        if hour and hour % settings.assay_interval == 0:
            scale = np.sqrt(FLOORS**2 + (0.1 * abs(truth[-1])) ** 2) * settings.noise
            measured = np.maximum(0, truth[-1] + rng.normal(0, scale))
            measured[1] = max(measured[1], measured[0])
            pending.append((hour + settings.delay, hour, measured))
            samples.append(
                dict(hour=hour, available_at=hour + settings.delay, values=measured.tolist())
            )
        arrived = [item for item in pending if item[0] <= hour]
        for _, sample, measurement in arrived:
            delivered[sample] = measurement
        pending = [item for item in pending if item[0] > hour]
        for name, observer in observers.items():
            observer.history[hour] = (observer.state.copy(), observer.factor, observer.maintenance)
            if arrived and name != "Frozen":
                observer.update(hour, delivered, actual_events, run.strength)
            estimates[name].append(observe(observer.state))
        active = observers[
            "Adaptive" if adaptive else ("State correction" if controlled else "Frozen")
        ]
        factors.append(active.factor)
        maintenance_factors.append(active.maintenance)
        if hour < steps:
            net = np.log(target[hour + 1] / target[hour]) if target is not None else 0
            checks.append(
                reachability(active.state, replace(p, mu_max=p.mu_max * active.factor), net)
            )
            if controlled:
                actual_events[hour, 1] = 0
                if hour % settings.interval == 0:
                    horizon = min(settings.horizon, steps - hour)
                    future = actual_events[hour : hour + horizon].copy()
                    future[:, 1] = 0
                    dose, record = choose_dose(
                        active.state,
                        active.effective_parameters(),
                        future,
                        run.strength,
                        target[hour + 1 : hour + horizon + 1],
                        settings,
                        active.factor,
                    )
                    actual_events[hour, 1] = dose / 1000
                    plan_start = hour
                    planned = np.array(record["planned_events"])
                    diagnostics.append(dict(hour=hour, **record))
            actions.append(actual_events[hour, 1] * 1000)
            post = event(state, actual_events[hour], run.strength)
            if (
                post[0] * 1000 > settings.max_volume_ml + 1e-6
                or observe(post)[2] > settings.max_glucose + 1e-6
            ):
                violations.append(
                    dict(
                        hour=hour,
                        volume_ml=float(post[0] * 1000),
                        glucose_gL=float(observe(post)[2]),
                    )
                )
            # Forecasts are issued at assay-result times, before seeing future measurements.
            if arrived or hour == 0:
                horizon = min(max(48, settings.assay_interval), steps - hour)
                inputs = (
                    planned[hour - plan_start : hour - plan_start + horizon]
                    if controlled
                    else actual_events[hour : hour + horizon]
                )
                for name, observer in observers.items():
                    effective = observer.effective_parameters()
                    pred = forecast(
                        observer.state, effective, inputs, run.strength, observer.factor
                    )
                    issued[name].append(
                        dict(
                            hour=hour,
                            values=pred.tolist(),
                            events=inputs.tolist(),
                            initial_state=observer.state.tolist(),
                            parameters=asdict(effective),
                            growth_factor=observer.factor,
                        )
                    )
                oracle = state.copy()
                values = [observe(oracle)]
                multipliers = []
                for k, amounts in enumerate(inputs, hour):
                    multiplier = (
                        settings.multiplier if disturbed and k >= settings.disturbance else 1.0
                    )
                    oracle = advance(oracle, p, amounts, run.strength, multiplier)
                    values.append(observe(oracle))
                    multipliers.append(multiplier)
                issued["Oracle"].append(
                    dict(
                        hour=hour,
                        values=np.array(values).tolist(),
                        events=inputs.tolist(),
                        initial_state=state.tolist(),
                        parameters=asdict(p),
                        growth_factors=multipliers,
                    )
                )
            state = advance(state, p, actual_events[hour], run.strength, actual_factor, step=0.5)
            for observer in observers.values():
                observer.state = advance(
                    observer.state,
                    observer.effective_parameters(),
                    actual_events[hour],
                    run.strength,
                    observer.factor,
                )
    truth = np.array(truth)
    scores = []
    for name, forecasts in issued.items():
        for lead in [*sorted({settings.assay_interval, 24, 48}), None]:
            errors = []
            for issued_forecast in forecasts:
                hour = issued_forecast["hour"]
                offset = (
                    lead
                    if lead is not None
                    else settings.assay_interval - hour % settings.assay_interval
                )
                if hour >= settings.disturbance and len(issued_forecast["values"]) > offset:
                    errors.append(
                        np.array(issued_forecast["values"][offset][:4]) - truth[hour + offset, :4]
                    )
            scores.append(
                dict(
                    model=name,
                    lead_h=lead,
                    count=len(errors),
                    rmse=np.sqrt(np.mean(np.square(errors), axis=0)).tolist() if errors else None,
                )
            )
    return dict(
        truth=truth.tolist(),
        estimates={k: np.array(v).tolist() for k, v in estimates.items()},
        factors=factors,
        maintenance_factors=maintenance_factors,
        estimated_factors={
            k: [item["factor"] for item in o.diagnostics] for k, o in observers.items()
        },
        updates={k: o.diagnostics for k, o in observers.items()},
        forecasts=issued,
        feed_ml=actions,
        events=actual_events.tolist(),
        strength=run.strength,
        assays=samples,
        violations=violations,
        volume_ml=None,
        controller=diagnostics,
        reachable=checks,
        scores=scores,
        settings=asdict(settings),
    )


def experiment(run, p, settings, target=None, mode="adaptation"):
    if mode not in ("control", "adaptation"):
        raise ValueError("Choose control or adaptation.")
    if mode == "control" and (target is None or len(target) != len(run.events)):
        raise ValueError("Provide an hourly VCD target covering the scenario.")
    names = (
        (
            ("Recorded recipe", False, False, False),
            ("Nominal MPC", True, False, False),
            ("Disturbed MPC", True, False, True),
            ("Adaptive MPC", True, True, True),
        )
        if mode == "control"
        else (("Growth disturbance", False, True, True),)
    )
    results = {
        name: trial(run, p, settings, target, control, adapt, disturb)
        for name, control, adapt, disturb in names
    }
    for item in results.values():
        volumes = initialize(run)[0] + np.cumsum(
            np.r_[
                0,
                np.array(item["feed_ml"]) / 1000
                + run.events[:-1, 2:].sum(axis=1)
                - run.events[:-1, 0],
            ]
        )
        item["volume_ml"] = (volumes * 1000).tolist()
        if mode == "control":
            item[
                "scores"
            ] = []  # Future feedback actions differ from issued planned-feed forecasts.
        if target is not None:
            item["tracking_rmse"] = float(
                np.sqrt(np.mean((np.array(item["truth"])[:, 0] - target) ** 2))
            )
    return dict(
        mode=mode,
        target=target.tolist() if target is not None else None,
        cases=results,
        parameters=asdict(p),
        note="Synthetic demonstration; no real-process control validation.",
    )

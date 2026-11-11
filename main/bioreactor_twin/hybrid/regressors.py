"""Small integral regressors, including a finite-rank Matérn Gaussian process."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import LinearConstraint, minimize

from bioreactor_twin.data.dataset import condition_size
from bioreactor_twin.hybrid.operators import boundary, integrate
from bioreactor_twin.hybrid.species import FEATURES, FLOORS, assays


def matern(x, centers, length):
    radius = np.sqrt(np.sum(((x[:, None, :] - centers[None, :, :]) / length) ** 2, axis=2))
    distance = np.sqrt(5) * radius
    return (1 + distance + distance**2 / 3) * np.exp(-distance)


def scaler(nodes, features):
    indices = []
    removed = []
    for name in features:
        index = FEATURES.index(name)
        if np.std(nodes[:, index]) < 1e-10:
            removed.append(name)
        elif any(abs(np.corrcoef(nodes[:, index], nodes[:, j])[0, 1]) > 0.95 for j in indices):
            removed.append(name)
        else:
            indices.append(index)
    mean = nodes[:, indices].mean(axis=0)
    std = nodes[:, indices].std(axis=0)
    return indices, mean, std, removed


class InvalidRateTrajectory(ValueError):
    def __init__(self, species, hour, concentration, sd):
        super().__init__(f"{species}: negative or non-finite concentration at hour {hour}.")
        self.concentration = concentration
        self.sd = sd


def limit_consumption(trace, concentration, start):
    """Limit removal to available mass and carry the resulting mass correction forward."""
    correction = 0.0
    first_limited = len(concentration)
    for hour in range(start, len(concentration)):
        amount = concentration[hour] * trace.volume[hour] + correction
        if amount < 0:
            first_limited = min(first_limited, hour)
            correction -= amount
            amount = 0.0
        concentration[hour] = amount / trace.volume[hour]
        correction *= trace.retention[hour]
    return first_limited


@dataclass
class RateModel:
    species: str
    family: str
    features: list
    mean: list
    std: list
    centers: list
    root: list
    length: float
    penalty: float
    scale: float
    coefficients: list
    covariance: list
    removed: list
    limit_depletion: bool = False

    def basis(self, nodes):
        indices = [FEATURES.index(name) for name in self.features]
        x = (nodes[:, indices] - self.mean) / self.std
        if self.family == "gp" and indices:
            kernel = matern(x, np.asarray(self.centers), self.length)
            x = solve_triangular(np.asarray(self.root), kernel.T, lower=True).T
        return np.column_stack((np.ones(len(nodes)), x)) * self.scale

    def predict(self, trace, *, diagnostics=None):
        design = integrate(trace, self.basis(trace.nodes))
        concentration = boundary(trace, self.species) + design @ self.coefficients
        start = int(trace.run.times[0])
        sd = None
        if self.family == "gp" and self.covariance:
            variance = np.einsum("ij,jk,ik->i", design, np.array(self.covariance), design)
            sd = np.sqrt(np.maximum(variance, 0))
        if self.limit_depletion and np.all(np.isfinite(concentration[start:])):
            first_limited = limit_consumption(trace, concentration, start)
            if diagnostics is not None and first_limited < len(concentration):
                diagnostics[self.species] = first_limited
            if sd is not None:
                sd[first_limited:] = np.nan
        invalid = np.flatnonzero(
            ~np.isfinite(concentration[start:]) | (concentration[start:] < -1e-9)
        )
        if invalid.size:
            hour = start + int(invalid[0])
            concentration[hour:] = np.nan
            if sd is not None:
                sd[hour:] = np.nan
            raise InvalidRateTrajectory(self.species, hour, concentration, sd)
        return concentration, sd

    def rates(self, trace):
        return self.basis(trace.nodes) @ self.coefficients

    def export(self):
        return asdict(self)


def fit_rate(traces, species, features=(), family="constant", penalty=0.001, length=1.0):
    nodes = np.concatenate([t.nodes for t in traces.values()])
    indices, mean, std, removed = scaler(nodes, features)
    names = [FEATURES[i] for i in indices]
    centers = []
    root = []
    if family == "gp" and indices:
        x = (nodes[:, indices] - mean) / std
        # Deterministic training-only landmarks. This is an explicit low-rank GP approximation.
        centers = x[np.linspace(0, len(x) - 1, min(16, len(x)), dtype=int)]
        root = cholesky(matern(centers, centers, length) + 1e-8 * np.eye(len(centers)), lower=True)
    model = RateModel(
        species,
        family,
        names,
        mean.tolist(),
        std.tolist(),
        np.asarray(centers).tolist(),
        np.asarray(root).tolist(),
        length,
        penalty,
        1.0,
        [],
        [],
        removed,
    )
    design, y, noise = [], [], []
    for trace in traces.values():
        run = trace.run
        times = run.times[1:]
        observed = assays(run, species)[1:]
        design.append(integrate(trace, model.basis(trace.nodes))[times])
        y.append(observed - boundary(trace, species)[times])
        count = condition_size(run, [t.run for t in traces.values()])
        noise.append(
            np.sqrt(FLOORS[species] ** 2 + (0.1 * abs(observed)) ** 2) * np.sqrt(len(times) * count)
        )
    x = np.concatenate(design)
    target = np.concatenate(y)
    sigma = np.concatenate(noise)
    model.scale = max(float(np.linalg.norm(target) / max(np.linalg.norm(x[:, 0]), 1e-12)), 1e-7)
    x = x * model.scale / sigma[:, None]
    target = target / sigma
    precision = x.T @ x + penalty * np.eye(x.shape[1])
    covariance = np.linalg.solve(precision, np.eye(len(precision)))
    model.coefficients = (covariance @ x.T @ target).tolist()
    model.covariance = covariance.tolist()
    enforce_training_positivity(model, traces, precision, x.T @ target)
    model.limit_depletion = True
    for trace in traces.values():
        model.predict(trace)
    return model


def enforce_training_positivity(model, traces, precision, linear):
    designs = []
    offsets = []
    for trace in traces.values():
        start = int(trace.run.times[0]) + 1
        designs.append(integrate(trace, model.basis(trace.nodes))[start:])
        offsets.append(boundary(trace, model.species)[start:])
    design = np.concatenate(designs)
    offset = np.concatenate(offsets)
    if np.min(offset + design @ model.coefficients) >= -1e-9:
        return
    fit = minimize(
        lambda w: 0.5 * w @ precision @ w - linear @ w,
        np.zeros(len(linear)),
        jac=lambda w: precision @ w - linear,
        constraints=[LinearConstraint(design, -offset, np.inf)],
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 150},
    )
    if not fit.success or np.min(offset + design @ fit.x) < -1e-8:
        raise ValueError(f"{model.species}: nonnegative rate fit failed.")
    model.coefficients = fit.x.tolist()
    # An unconstrained Gaussian covariance is inappropriate for this constrained fit.
    model.covariance = []

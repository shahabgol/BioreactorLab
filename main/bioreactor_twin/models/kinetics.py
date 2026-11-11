"""Small macroscopic Monod model; parameters have explicit units and bounds."""

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True)
class Parameters:
    mu_max: float = 0.05  # 1/h
    k_g: float = 3.0  # mM
    capacity: float = 28.0  # billion cells/L
    death: float = 0.0005  # 1/h
    death_density: float = 0.015  # 1/h
    glucose_growth: float = 1.5  # mmol/billion cells
    glucose_maintenance: float = 0.03  # mmol/billion cells/h
    glutamate_growth: float = 0.05  # mmol/billion cells
    glutamate_uptake: float = 0.001  # mmol/billion cells/h
    lactate_yield: float = 1.0  # mol/mol
    lactate_uptake: float = 0.02  # mmol/billion cells/h
    k_e: float = 0.1  # mM, fixed screening constant
    k_lactate: float = 3.0  # mM, fixed screening constant
    lysis: float = 0.0  # 1/h; fixed in the baseline

    def __post_init__(self):
        values = asdict(self)
        positive = ("mu_max", "k_g", "capacity", "k_e", "k_lactate")
        if not all(isfinite(value) and value >= 0 for value in values.values()):
            raise ValueError("Parameters must be finite and nonnegative")
        if any(values[key] <= 0 for key in positive):
            raise ValueError("Growth maximum, capacity and saturation constants must be positive")


def monod(concentration: float, half_saturation: float) -> float:
    """Nonnegative substrate saturation; callers reject material negative states."""
    c = max(concentration, 0.0)
    return c / (half_saturation + c)


def rates(state, p: Parameters):
    """Growth/death and signed metabolite rates from the current amounts."""
    volume, viable, dead, glucose, glutamate, lactate = state
    fg = monod(glucose / volume, p.k_g)
    fe = monod(glutamate / volume, p.k_e)
    fraction = (viable + dead) / volume / p.capacity
    growth = p.mu_max * fg * fe * max(0.0, 1.0 - fraction)
    death = p.death + p.death_density * fraction**2
    uptake = p.glucose_growth * growth + p.glucose_maintenance * fg
    glutamate_rate = p.glutamate_growth * growth - p.glutamate_uptake * fe
    lactate_rate = p.lactate_yield * uptake - p.lactate_uptake * (
        monod(lactate / volume, p.k_lactate) * (1 - growth / p.mu_max)
    )
    return growth, death, uptake, glutamate_rate, lactate_rate


def derivative(state, p: Parameters):
    """Death transfers cells; only lysis removes dead cells from total counts."""
    _, viable, dead, _, _, _ = state
    growth, death, glucose, glutamate, lactate = rates(state, p)
    return (
        0.0,
        (growth - death) * viable,
        death * viable - p.lysis * dead,
        -glucose * viable,
        glutamate * viable,
        lactate * viable,
    )

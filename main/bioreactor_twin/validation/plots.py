"""Independent plotting layer; consumes frozen simulations and observed assays."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bioreactor_twin.data.dataset import NAMES
from bioreactor_twin.validation.uncertainty import envelope


def plot_output(runs, predictions, repeatability_sd, column, destination):
    """One figure per output, with separate run panels and explicit split labels."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    for axis, run in zip(axes.flat, runs.values(), strict=True):
        values = predictions[run.number]
        hours = np.arange(len(values))
        low, high = envelope(values, repeatability_sd)
        axis.fill_between(
            hours,
            low[:, column],
            high[:, column],
            color="#286b98",
            alpha=0.18,
            label="Descriptive repeatability band (±1.96 SD)",
        )
        axis.plot(hours, values[:, column], color="#16658c", lw=1.7, label="Model, pre-event")
        axis.scatter(
            run.times[1:],
            run.observed[1:, column],
            s=30,
            color="#bb4b23",
            label="Experiment (scored)",
            zorder=3,
        )
        axis.scatter(
            run.times[0],
            run.observed[0, column],
            marker="D",
            s=30,
            color="#555555",
            label="Initialization (not scored)",
        )
        split = "HELD OUT" if run.number in (1, 2) else "Calibration"
        axis.set_title(f"Run {run.number} · {split}")
        axis.set_xlabel("Elapsed time (h)")
        axis.set_ylabel("Million cells/mL" if column < 2 else "g/L")
        axis.grid(alpha=0.2)
        axis.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"{NAMES[column]}: model and experiment", fontsize=16)
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    fig.savefig(destination, dpi=160)
    plt.close(fig)

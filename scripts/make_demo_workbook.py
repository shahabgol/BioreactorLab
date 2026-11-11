"""Create a synthetic workbook with three independent demo conditions."""

import argparse
from pathlib import Path

import numpy as np
from bioreactor_twin.data.audit import HEADERS
from bioreactor_twin.data.dataset import Run
from bioreactor_twin.models.kinetics import Parameters
from bioreactor_twin.simulation.nitrogen import Nitrogen, simulate_mechanistic
from openpyxl import Workbook


def rows(number, feed_b, start):
    duration = 144
    events = np.zeros((duration + 1, 4))
    assay_hours = np.arange(0, duration + 1, 24)
    events[assay_hours[1:], 0] = 0.0005
    events[[36, 72, 108], 1] = 0.0008
    events[[48, 96], 2] = feed_b / 1000
    volume = 0.18 + np.cumsum(events[:, 1:].sum(axis=1) - events[:, 0])
    run = Run(
        number,
        events,
        volume,
        np.array([0]),
        np.array([[start, start + 0.03, 4.8, 0.36, 0.17]]),
        (),
        additional={"GLN": np.array([0.04]), "AMM": np.array([0.01])},
    )
    prediction = simulate_mechanistic(run, Parameters(), Nitrogen(0.06, 0.003, 0.02))
    for hour, (sample, a, b, c) in enumerate(events * 1000):
        if hour in assay_hours:
            vcd, tcd, glc, glu, lac, gln, amm = prediction[hour]
            panel = ((tcd - vcd) * 1e6, tcd * 1e6, vcd * 1e6, glc)
            extra = (7.1, 7.1, gln, glu, lac, amm)
        else:
            panel, extra = (None,) * 4, (None,) * 6
        yield (number, hour, *panel, sample, a, b, volume[hour] * 1000, c, *extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/demo.xlsx"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. Choose a new filename.")
    book = Workbook()
    sheet = book.active
    sheet.title = "SampleData"
    sheet.append(HEADERS)
    for number, feed, start in [(101, 1.5, 0.42), (102, 2.5, 0.48), (103, 3.5, 0.53)]:
        for row in rows(number, feed, start):
            sheet.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    book.save(args.output)
    print(f"Synthetic demo saved to {args.output}")


if __name__ == "__main__":
    main()

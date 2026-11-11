"""Launch the local Dash workspace: python -m bioreactor_twin.web.app."""

import argparse
from pathlib import Path

from dash import Dash

from bioreactor_twin.web.callbacks import register_callbacks
from bioreactor_twin.web.layout import layout
from bioreactor_twin.web.workspace import ROOT, Lab


def create_app(root=ROOT, lab=None):
    app = Dash(
        __name__,
        title="Bioreactor lab · Digital twin",
        assets_folder=str(Path(__file__).parent / "assets"),
        update_title="Working…",
    )
    app.server.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024
    app.layout = layout
    app.lab = lab if lab is not None else Lab(root)
    app.twin_callbacks = register_callbacks(app, app.lab)
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args(argv)
    create_app().run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()

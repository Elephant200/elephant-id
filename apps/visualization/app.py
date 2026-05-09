"""Flask app factory and CLI entry point."""

from __future__ import annotations

import logging
import os

from flask import Flask

from .routes import create_blueprint
from .samples import reconcile_all_starred
from .state import ReviewerState


def create_app() -> Flask:
    """Build a fully configured Flask app.

    Performs startup work (CSV load + one-shot starred reconciliation) so the
    returned app is ready to serve. Raises if the dataset CSV is missing.
    """
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    state = ReviewerState()
    state.load()
    reconcile_all_starred()

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.register_blueprint(create_blueprint(state))
    app.extensions["reviewer_state"] = state
    return app


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    app = create_app()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

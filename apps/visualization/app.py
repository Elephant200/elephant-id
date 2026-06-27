"""Flask app factory and CLI entry point."""

from __future__ import annotations

import logging
import os

from flask import Flask

from elephant_id.dataset import Dataset
from elephant_id.log import configure_logging

from .analyzer import AnalyzerWorkbench
from .config import CODED_ROOT, CSV_PATH
from .routes import create_blueprint
from .sam3 import Sam3Workbench
from .samples import reconcile_all_starred
from .state import ReviewerState

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Build a fully configured Flask app.

    Performs startup work so the returned app is ready to serve. Raises
    if the dataset CSV is missing.
    """
    configure_logging()  # loguru, for elephant_id library logs
    logging.basicConfig(  # stdlib, for this dev app's own logs
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dataset = Dataset(dataset_root=CODED_ROOT, metadata_path=CSV_PATH)
    state = ReviewerState()
    state.load(dataset)
    reconcile_all_starred()
    analyzer = AnalyzerWorkbench(dataset)
    sam3 = Sam3Workbench(dataset)

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.register_blueprint(
        create_blueprint(state, dataset=dataset, analyzer=analyzer, sam3=sam3)
    )
    app.extensions["reviewer_state"] = state
    app.extensions["dataset"] = dataset
    app.extensions["analyzer_workbench"] = analyzer
    app.extensions["sam3_workbench"] = sam3
    return app


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    app = create_app()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

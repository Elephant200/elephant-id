"""Flask app factory and CLI entry point."""

from __future__ import annotations

import logging
import os

from flask import Flask

from elephant_id.dataset import Dataset

from .config import CODED_ROOT, CSV_PATH
from .routes import create_blueprint
from .samples import reconcile_all_starred
from .state import ReviewerState

logger = logging.getLogger(__name__)


def _build_sam3_service(dataset: Dataset):
    """Construct the Sam3Service if dependencies and API key are available.

    Returns None when the optional ``local`` extras (inference-sdk) are not
    installed or ``ROBOFLOW_API_KEY`` is unset. The SAM3 route surfaces a 503
    in that case.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv()

    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        logger.info("ROBOFLOW_API_KEY not set; SAM3 endpoint disabled.")
        return None

    try:
        from elephant_id.ai import Sam3Service
    except ImportError:
        logger.warning(
            "inference-sdk not installed; SAM3 endpoint disabled. "
            "Install with `uv sync --extra local`."
        )
        return None

    return Sam3Service(api_key=api_key, dataset=dataset)


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

    dataset = Dataset(dataset_root=CODED_ROOT, metadata_path=CSV_PATH)
    sam3 = _build_sam3_service(dataset)

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.register_blueprint(create_blueprint(state, dataset=dataset, sam3=sam3))
    app.extensions["reviewer_state"] = state
    app.extensions["dataset"] = dataset
    app.extensions["sam3"] = sam3
    return app


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    app = create_app()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

"""Flask app factory and routes for the lightweight image picker."""

from __future__ import annotations

import io
import os

from flask import Flask, abort, jsonify, render_template, request, send_file

from elephant_id.dataset import Dataset
from elephant_id.log import configure_logging

from .catalog import PhotoCatalog
from .config import CODED_ROOT, CSV_PATH
from .model import CandidateAnalyzer
from .state import PickerState


def create_app() -> Flask:
    """Build the configured image picker Flask app."""
    configure_logging()
    dataset = Dataset(dataset_root=CODED_ROOT, metadata_path=CSV_PATH)
    catalog = PhotoCatalog.from_paths()
    analyzer = CandidateAnalyzer(dataset)
    state = PickerState(dataset=dataset, catalog=catalog, analyzer=analyzer)

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.extensions["picker_state"] = state

    @app.get("/")
    def index():
        """Render the single-page picker UI."""
        return render_template("index.html")

    @app.get("/api/state")
    def api_state():
        """Return global picker state."""
        return jsonify(state.view())

    @app.get("/api/identity")
    def api_identity():
        """Return candidates for one side/identity."""
        side = (request.args.get("side") or "").strip()
        identity = (request.args.get("identity") or "").strip()
        try:
            if not identity:
                identity = state.current_identity(side) or ""
            if not identity:
                return jsonify({"error": "No identity"}), 404
            return jsonify(state.identity_payload(side, identity))
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

    @app.post("/api/nav")
    def api_nav():
        """Navigate one side queue."""
        data = _json_body()
        try:
            side = str(data.get("side") or "")
            delta = int(data.get("delta") or 0)
            return jsonify(state.navigate(side, delta))
        except (TypeError, ValueError) as error:
            return jsonify({"error": str(error)}), 400

    @app.post("/api/select")
    def api_select():
        """Toggle one candidate checkbox."""
        data = _json_body()
        try:
            return jsonify(
                state.toggle_selection(
                    side=str(data.get("side") or ""),
                    identity=str(data.get("identity") or ""),
                    candidate_id=str(data.get("candidateId") or ""),
                    selected=bool(data.get("selected")),
                )
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

    @app.post("/api/done")
    def api_done():
        """Export selected originals for one identity."""
        data = _json_body()
        try:
            return jsonify(
                state.mark_done(
                    side=str(data.get("side") or ""),
                    identity=str(data.get("identity") or ""),
                )
            )
        except (OSError, RuntimeError, ValueError) as error:
            return jsonify({"error": str(error)}), 400

    @app.get("/api/crop")
    def api_crop():
        """Return one crop preview image."""
        side = (request.args.get("side") or "").strip()
        identity = (request.args.get("identity") or "").strip()
        candidate_id = (request.args.get("candidateId") or "").strip()
        try:
            jpeg = state.crop_jpeg(side, identity, candidate_id)
        except (RuntimeError, ValueError, KeyError):
            abort(404)
        return send_file(io.BytesIO(jpeg), mimetype="image/jpeg")

    return app


def main() -> None:
    """Run the development server."""
    port = int(os.environ.get("PORT", "8010"))
    app = create_app()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


def _json_body() -> dict:
    """Return a JSON request body or an empty dict."""
    return request.get_json(force=True, silent=True) or {}

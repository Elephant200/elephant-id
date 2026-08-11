"""Flask app factory and routes for the matching image picker."""

from __future__ import annotations

import io
import os

from flask import Flask, abort, jsonify, render_template, request, send_file
from flask.typing import ResponseReturnValue

from elephant_id.dataset import Dataset
from elephant_id.log import configure_logging

from .analysis import CandidateAnalyzer
from .catalog import PhotoCatalog
from .config import CODED_ROOT, CSV_PATH
from .manifest import ManifestStore
from .state import PickerState


def create_app() -> Flask:
    """Build the configured matching image picker Flask app."""
    configure_logging()
    dataset = Dataset(dataset_root=CODED_ROOT, metadata_path=CSV_PATH)
    catalog = PhotoCatalog.from_dataset(dataset)
    analyzer = CandidateAnalyzer(dataset)
    manifest = ManifestStore(dataset)
    state = PickerState(
        dataset=dataset, catalog=catalog, analyzer=analyzer, manifest=manifest
    )

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.extensions["picker_state"] = state

    @app.get("/")
    def index() -> ResponseReturnValue:
        """Render the single-page picker UI."""
        return render_template("index.html")

    @app.get("/api/elephants")
    def api_elephants() -> ResponseReturnValue:
        """Return eligible elephants and scan progress."""
        return jsonify(state.elephants_view())

    @app.get("/api/elephant/<identity>")
    def api_elephant(identity: str) -> ResponseReturnValue:
        """Return one elephant's qualifying sightings and candidates."""
        try:
            return jsonify(state.elephant_view(identity))
        except KeyError:
            abort(404)

    @app.post("/api/pick")
    def api_pick() -> ResponseReturnValue:
        """Record a candidate as a sighting's canonical pick for one side."""
        data = request.get_json(force=True, silent=True) or {}
        try:
            return jsonify(
                state.record_pick(
                    identity=str(data.get("identity") or ""),
                    sighting_date=str(data.get("sightingDate") or ""),
                    side=str(data.get("side") or ""),
                    candidate_id=str(data.get("candidateId") or ""),
                )
            )
        except (KeyError, ValueError) as error:
            return jsonify({"error": str(error)}), 400

    @app.get("/api/crop")
    def api_crop() -> ResponseReturnValue:
        """Return one candidate's ear crop as a JPEG preview."""
        try:
            jpeg = state.crop_jpeg(
                identity=(request.args.get("identity") or "").strip(),
                sighting_date=(request.args.get("sightingDate") or "").strip(),
                side=(request.args.get("side") or "").strip(),
                candidate_id=(request.args.get("candidateId") or "").strip(),
            )
        except (KeyError, ValueError, RuntimeError):
            abort(404)
        return send_file(io.BytesIO(jpeg), mimetype="image/jpeg")

    return app


def main() -> None:
    """Run the development server."""
    port = int(os.environ.get("PORT", "8010"))
    app = create_app()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)

"""HTTP routes for the sighting reviewer.

All endpoints return JSON except for thumbnail endpoints (binary images) and
``/`` (the HTML shell). The frontend in ``static/app.js`` is the sole client;
field names here form the API contract with it.
"""

from __future__ import annotations

from flask import Blueprint, abort, jsonify, render_template, request, send_file

from .config import PAGE_SIZE_DEFAULT, THUMB_MAX_SIZE
from .filters import FilterConfig
from .paths import safe_saved_sighting_dir, safe_saved_sighting_file
from .samples import first_priority_or_any_image
from .state import ReviewerState, list_saved_sighting_entries
from .thumbs import absolute_thumb, coded_thumb


def create_blueprint(state: ReviewerState) -> Blueprint:
    bp = Blueprint("reviewer", __name__)

    def _json_body() -> dict:
        return request.get_json(force=True, silent=True) or {}

    @bp.get("/")
    def index():
        return render_template("index.html")

    # -------- queue navigation -----------------------------------------

    @bp.get("/api/state")
    def api_state():
        return jsonify(state.view())

    @bp.post("/api/nav")
    def api_nav():
        delta = int(_json_body().get("delta", 0))
        if delta:
            state.nav(delta)
        return jsonify(state.view())

    @bp.post("/api/page")
    def api_page():
        delta = int(_json_body().get("delta", 0))
        if delta:
            state.page_nav(delta)
        return jsonify(state.view())

    @bp.post("/api/page_size")
    def api_page_size():
        page_size = int(_json_body().get("pageSize", PAGE_SIZE_DEFAULT))
        state.set_page_size(page_size)
        return jsonify(state.view())

    @bp.post("/api/shuffle")
    def api_shuffle():
        state.set_shuffle(bool(_json_body().get("enabled")))
        return jsonify(state.view())

    @bp.post("/api/filter")
    def api_filter():
        state.apply_filter(FilterConfig.from_json(_json_body()))
        return jsonify(state.view())

    @bp.post("/api/elephant_only")
    def api_elephant_only():
        state.elephant_only_set(bool(_json_body().get("enabled")))
        return jsonify(state.view())

    # -------- mutations -----------------------------------------------

    @bp.post("/api/toggle_priority_image")
    def api_toggle_priority_image():
        data = _json_body()
        samples_rel = (data.get("samplesRel") or "").strip()
        if samples_rel:
            state.toggle_priority_samples_file(samples_rel)
        else:
            rel = (data.get("imagePath") or "").strip()
            if rel:
                state.toggle_priority_image(rel)
        return jsonify(state.view())

    @bp.post("/api/undo")
    def api_undo():
        state.undo()
        return jsonify(state.view())

    # -------- saved (starred) browser ---------------------------------

    @bp.get("/api/saved/list")
    def api_saved_list():
        return jsonify(state.saved_list_dict())

    @bp.get("/api/saved/sighting_images")
    def api_saved_sighting_images():
        rel = (request.args.get("rel") or "").strip()
        try:
            return jsonify({"rels": list_saved_sighting_entries(rel)})
        except ValueError:
            return jsonify({"error": "invalid rel"}), 400

    @bp.post("/api/saved/remove")
    def api_saved_remove():
        data = _json_body()
        kind = (data.get("kind") or "").strip()
        if kind != "sighting":
            return jsonify({"error": "invalid kind"}), 400
        try:
            state.saved_remove_sighting((data.get("rel") or "").strip())
        except (ValueError, FileNotFoundError, OSError) as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"review": state.view(), "saved": state.saved_list_dict()})

    # -------- thumbnails ----------------------------------------------

    @bp.get("/api/saved/file_thumb")
    def api_saved_file_thumb():
        rel = (request.args.get("rel") or "").strip()
        size = int(request.args.get("s", THUMB_MAX_SIZE))
        try:
            path = safe_saved_sighting_file(rel)
            return send_file(absolute_thumb(path, "sf", size))
        except ValueError:
            abort(404)

    @bp.get("/api/saved/thumb")
    def api_saved_thumb():
        kind = (request.args.get("kind") or "sighting").strip()
        size = int(request.args.get("s", THUMB_MAX_SIZE))
        if kind != "sighting":
            abort(404)
        rel = (request.args.get("rel") or "").strip()
        try:
            folder = safe_saved_sighting_dir(rel)
        except ValueError:
            abort(404)
        first = first_priority_or_any_image(folder)
        if first is None:
            abort(404)
        return send_file(absolute_thumb(first, "sig", size))

    @bp.get("/thumb")
    def thumb():
        rel = request.args.get("p", "")
        size = int(request.args.get("s", THUMB_MAX_SIZE))
        try:
            return send_file(coded_thumb(rel, size))
        except (FileNotFoundError, ValueError):
            abort(404)

    return bp

"""Run the local persistent sighting ear-pair review workspace.

Run `uv run --group visualization python scripts/curation/review_pairs.py`.
"""

import argparse
import csv
import json
import mimetypes
import threading
from pathlib import Path
from typing import Any
from uuid import UUID

from flask import Flask, Response, current_app, jsonify, request, send_file

from elephant_id.curation import Candidate, ReviewSession
from elephant_id.dataset import Dataset

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_RESULTS = ROOT / "outputs/curation/archive/results.json"
ARCHIVE_CROPS = ARCHIVE_RESULTS.parent / "images"
DATASET_ROOT = ROOT / "dataset/elephants-alive/coded"
METADATA = ROOT / "dataset/elephants-alive/images.csv"
REVIEW_ROOT = ROOT / "outputs/curation/sighting-pair-review"


def load_session(
    results_path: Path = ARCHIVE_RESULTS,
    dataset_root: Path = DATASET_ROOT,
    metadata_path: Path = METADATA,
    review_root: Path = REVIEW_ROOT,
) -> ReviewSession:
    """Build a review session from frozen assessments and dataset metadata."""
    dataset = Dataset(dataset_root, metadata_path)
    source_paths: dict[UUID, Path] = {}
    photos_by_elephant: dict[str, list[UUID]] = {}
    with metadata_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            photo_id = UUID(row["photo_id"])
            source_paths[photo_id] = dataset_root / row["image_path"]
            photos_by_elephant.setdefault(row["name"], []).append(photo_id)

    candidates: list[Candidate] = []
    for photo_record in json.loads(results_path.read_text()):
        photo_id = UUID(photo_record["photo_id"])
        photo = dataset.photo(photo_id)
        elephant = dataset.known_elephant_name(photo.sighting_id)
        if str(photo.sighting_id) != photo_record["sighting_id"] or elephant != photo_record["identity"]:
            raise ValueError(f"Assessment metadata conflicts with Dataset for photo {photo_id}")
        for ear in photo_record.get("ears", ()):
            assessment = ear.get("assessment") or {}
            score = (assessment.get("ear") or {}).get("reid_quality")
            if not isinstance(score, int):
                continue
            crop_path = results_path.parent / ear["preview"]
            side = ear.get("side") if ear.get("side") in ("left", "right") else "unresolved"
            candidates.append(
                Candidate(
                    crop_key=ear["input_key"],
                    photo_id=photo_id,
                    sighting_id=photo.sighting_id,
                    elephant=elephant,
                    side=side,
                    score=score,
                    crop_box=tuple(ear["box"]),
                    crop_path=str(crop_path),
                    original_path=str(source_paths[photo_id]),
                    assessment_provenance=f"{results_path}:{ear['input_key']}",
                    available=crop_path.is_file(),
                )
            )
    return ReviewSession(
        tuple(candidates),
        review_root / "state.json",
        review_root / "approved-manifest.csv",
        {name: tuple(photo_ids) for name, photo_ids in photos_by_elephant.items()},
        dataset.photo_store,
    )


def create_app(session: ReviewSession, metadata_path: Path = METADATA) -> Flask:
    """Create the thin local Flask application around a review session."""
    app = Flask(__name__)
    with metadata_path.open(newline="") as handle:
        app.config["SIGHTING_DATES"] = {
            row["sighting_id"]: row["date"] for row in csv.DictReader(handle)
        }
    lock = threading.Lock()

    @app.get("/")
    def index() -> str:
        """Serve the side-at-a-time review workspace."""
        return _PAGE

    @app.get("/api/screen")
    def screen() -> Response:
        """Return the current durable review screen."""
        return jsonify(_web_screen(session.as_dict()))

    @app.post("/api/select")
    def select() -> tuple[Response, int] | Response:
        """Select a crop and atomically advance the review."""
        return _mutate(lock, lambda: session.select(_json_body()["crop_key"]), session)

    @app.post("/api/reject")
    def reject() -> tuple[Response, int] | Response:
        """Reject the current sighting and advance."""
        return _mutate(lock, session.reject, session)

    @app.post("/api/back")
    def back() -> tuple[Response, int] | Response:
        """Navigate to the previous actual review screen."""
        return _mutate(lock, session.back, session)

    @app.post("/api/scroll")
    def scroll() -> tuple[Response, int] | Response:
        """Persist the current screen scroll offset."""
        try:
            with lock:
                body = _json_body()
                session.save_scroll(UUID(body["sighting_id"]), body["side"], int(body["scroll_y"]))
            return jsonify({"saved": True})
        except (KeyError, TypeError, ValueError, OSError) as error:
            return jsonify({"error": str(error)}), 400

    @app.get("/api/export")
    def export() -> Response:
        """Download the portable approved-pair and exclusion export."""
        return jsonify(session.export())

    @app.post("/api/restore")
    def restore() -> tuple[Response, int] | Response:
        """Validate and restore a portable review backup."""
        return _mutate(lock, lambda: session.restore(_json_body()), session)

    @app.get("/crop/<crop_key>")
    def crop(crop_key: str) -> Response:
        """Serve one frozen crop by inventory key."""
        candidate = next((item for item in session.screen().candidates if item.crop_key == crop_key), None)
        if candidate is None or not candidate.available:
            return Response("Crop unavailable", status=404)
        return send_file(candidate.crop_path)

    @app.get("/original/<crop_key>")
    def original(crop_key: str) -> Response:
        """Serve original sighting context for a visible crop."""
        candidate = next((item for item in session.screen().candidates if item.crop_key == crop_key), None)
        if candidate is None or not candidate.original_path:
            return Response("Original unavailable", status=404)
        try:
            data = session.read_original(crop_key)
        except (FileNotFoundError, KeyError):
            return Response("Original unavailable", status=404)
        media_type = mimetypes.guess_type(candidate.original_path)[0] or "application/octet-stream"
        return Response(data, mimetype=media_type)

    return app


def _json_body() -> dict[str, Any]:
    """Return an object JSON body or raise a useful validation error."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")
    return body


def _mutate(lock: threading.Lock, action: Any, session: ReviewSession) -> tuple[Response, int] | Response:
    """Serialize a session mutation and expose persistence errors."""
    try:
        with lock:
            action()
        return jsonify(_web_screen(session.as_dict()))
    except (KeyError, TypeError, ValueError, OSError) as error:
        return jsonify({"error": str(error)}), 400


def _web_screen(value: dict[str, object]) -> dict[str, object]:
    """Replace private filesystem paths with local HTTP routes."""
    value["sighting_date"] = current_app.config["SIGHTING_DATES"].get(value["sighting_id"])
    for candidate in value["candidates"]:
        candidate["crop_path"] = f"/crop/{candidate['crop_key']}"
        candidate["original_path"] = f"/original/{candidate['crop_key']}"
    return value


def main() -> None:
    """Load frozen inventory and run the localhost review server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    args = parser.parse_args()
    create_app(load_session()).run(host=args.host, port=args.port, debug=False)


_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlphaPhant sighting pair review</title><style>
:root{color-scheme:dark;--ink:#f4f0e7;--muted:#a7afa9;--line:#38423d;--accent:#d8a85e;--panel:#18201c}
*{box-sizing:border-box}body{margin:0;background:#101512;color:var(--ink);font:16px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:4;display:flex;align-items:center;gap:16px;min-height:72px;padding:12px 22px;background:#101512ee;border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}
h1{font:600 19px/1.2 Georgia,serif;margin:0}.meta{color:var(--muted);flex:1}.progress{font-weight:700;color:var(--accent)}button,a.inspect{border:1px solid #667269;border-radius:7px;background:#202a25;color:var(--ink);padding:9px 13px;text-decoration:none;cursor:pointer}button:hover,a.inspect:hover{border-color:var(--accent)}button:disabled{opacity:.45;cursor:wait}
#error{display:none;position:sticky;top:72px;z-index:5;background:#7c2929;padding:12px 22px}.score{padding:22px 0 8px;border-bottom:1px solid var(--line);font:600 17px Georgia,serif;color:var(--accent)}
main{max-width:2240px;margin:auto;padding:0 24px 90px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));column-gap:24px;align-items:start;position:relative}.empty{grid-column:1/-1}.score-dividers{position:absolute;inset:0;pointer-events:none;overflow:visible;z-index:1}.candidate{min-width:0;margin:18px 0 34px;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}.candidate.selected{border:2px solid var(--accent)}
.choose{display:block;width:100%;padding:0;border:0;border-radius:0;background:#090c0a}.choose img{display:block;margin:auto;max-width:100%;width:auto;min-height:420px;max-height:78vh;object-fit:contain}.choose.unavailable{min-height:420px;color:var(--muted)}
.caption{display:flex;gap:14px;align-items:center;padding:11px 14px;color:var(--muted)}.caption .inspect{margin-left:auto}.empty{max-width:680px;margin:18vh auto;text-align:center}.empty h2{font:600 34px Georgia,serif}
@media(max-width:650px){header{flex-wrap:wrap}.meta{order:3;flex-basis:100%}main{padding-inline:10px;column-gap:24px}.choose img,.choose.unavailable{min-height:300px}.caption{font-size:13px;flex-wrap:wrap}}
</style></head><body><header><button id="back">← Back</button><div><h1 id="title">Loading review…</h1><div class="meta" id="meta"></div></div><div class="meta"></div><div class="progress" id="progress"></div><button id="reject">Reject sighting</button><a id="export" href="/api/export" target="_blank" download>Backup</a><button id="restore">Restore</button><input id="restore-file" type="file" accept="application/json" hidden></header><div id="error"></div><main id="content"></main>
<script>
'use strict';const $=id=>document.getElementById(id);let busy=false,scrollTimer,current=null;
async function call(path,body={}){if(busy)return;clearTimeout(scrollTimer);busy=true;document.querySelectorAll('button').forEach(x=>x.disabled=true);$('error').style.display='none';try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),data=await r.json();if(!r.ok)throw Error(data.error||'Save failed');render(data)}catch(e){$('error').textContent=`Not saved: ${e.message}`;$('error').style.display='block'}finally{busy=false;document.querySelectorAll('button:not(.choose),button.choose.loaded').forEach(x=>x.disabled=false)}}
function render(s){current=s;if(s.status==='complete'){$('title').textContent='Review queue complete';$('meta').textContent='All active score 10, 9, and 8 sightings are decided.';$('progress').textContent='';$('reject').hidden=true;$('content').innerHTML='<section class="empty"><h2>Queue complete</h2><p>Your accepted pairs are saved. Back remains available for revisions.</p></section>';return}$('reject').hidden=false;$('title').textContent=`${s.elephant} · ${s.sighting_date||"Date unavailable"} · ${s.side} ear`;$('meta').textContent=`Proposed tier ${s.tier} · ${s.candidates.length} assessed candidates · click an image to ${s.side==='left'?'continue':'save pair'}`;$('progress').textContent=`${s.approved_count} / 5 approved`;
current=s;let html='';for(const c of s.candidates){html+=`<article data-score="${c.score}" class="candidate ${c.crop_key===s.selected_crop_key?'selected':''}"><button class="choose ${c.available?'':'unavailable'}" data-key="${c.crop_key}" disabled>${c.available?`<img src="${c.crop_path}" alt="${s.side} ear candidate, score ${c.score}" loading="lazy">`:'Image unavailable'}</button><div class="caption"><span>Score ${c.score}</span><span>Photo ${c.photo_id.slice(0,8)}</span><a class="inspect" href="${c.crop_path}" target="_blank">Enlarge crop</a><a class="inspect" href="${c.original_path}" target="_blank">Original context</a></div></article>`}$('content').innerHTML=html+'<svg class="score-dividers" aria-hidden="true"></svg>';drawDividers();document.querySelectorAll('.choose').forEach(x=>{const img=x.querySelector('img');if(img){img.onload=()=>{x.classList.add('loaded');if(!busy)x.disabled=false};img.onerror=()=>{x.classList.add('unavailable');x.textContent='Image unavailable'}}x.onclick=()=>call('/api/select',{crop_key:x.dataset.key})});requestAnimationFrame(()=>scrollTo(0,s.scroll_y||0))}
function drawDividers(){const grid=$('content'),svg=grid.querySelector('.score-dividers');if(!svg)return;const bounds=grid.getBoundingClientRect(),cards=[...grid.querySelectorAll('.candidate')],boxes=cards.map(c=>{const r=c.getBoundingClientRect();return {left:r.left-bounds.left,right:r.right-bounds.left,top:r.top-bounds.top,bottom:r.bottom-bounds.top}});svg.setAttribute('width',bounds.width);svg.setAttribute('height',grid.scrollHeight);let paths='';for(let i=0;i<cards.length;i++){if(i&&cards[i].dataset.score===cards[i-1].dataset.score)continue;const b=boxes[i],left=boxes[0].left,right=boxes.length>1?boxes[1].right:bounds.width-left;let d;if(i%2===0){const followsStep=i>=2&&cards[i-2].dataset.score!==cards[i-1].dataset.score,mid=boxes.length>1?(boxes[0].right+boxes[1].left)/2:left;d=`M ${followsStep?mid:left} ${Math.max(4,b.top-26)+(followsStep?6:0)} H ${right}`}else{const a=boxes[i-1],mid=(a.right+b.left)/2,top=Math.max(4,Math.min(a.top,b.top)-26),anotherBoundary=i+1<cards.length&&cards[i+1].dataset.score!==cards[i].dataset.score,bottom=Math.max(a.bottom,b.bottom)+26-(anotherBoundary?6:0);d=`M ${right} ${top} H ${mid} V ${bottom} H ${left}`}paths+=`<path d="${d}" fill="none" stroke="#d8a85e" stroke-width="2" stroke-linejoin="round"/>`}svg.innerHTML=paths}new ResizeObserver(drawDividers).observe($('content'));
$('back').onclick=()=>call('/api/back');$('reject').onclick=()=>call('/api/reject');$('restore').onclick=()=>$('restore-file').click();$('restore-file').onchange=async e=>{try{call('/api/restore',JSON.parse(await e.target.files[0].text()))}catch(err){$('error').textContent=`Restore failed: ${err.message}`;$('error').style.display='block'}};document.addEventListener('keydown',e=>{if(e.repeat||e.altKey||e.ctrlKey||e.metaKey||['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowLeft'){e.preventDefault();call('/api/back')}else if(e.key==='ArrowRight'&&current?.status==='review'&&current.selected_crop_key){e.preventDefault();call('/api/select',{crop_key:current.selected_crop_key})}});addEventListener('scroll',()=>{if(!current||current.status!=='review')return;clearTimeout(scrollTimer);const location={sighting_id:current.sighting_id,side:current.side,scroll_y:scrollY};scrollTimer=setTimeout(()=>fetch('/api/scroll',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(location)}),250)});fetch('/api/screen').then(r=>r.json()).then(render).catch(e=>{$('error').textContent=e.message;$('error').style.display='block'});
</script></body></html>'''


if __name__ == "__main__":
    main()

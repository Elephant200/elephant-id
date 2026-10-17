"""Run the local persistent sighting ear-pair review workspace.

Run `uv run --group visualization python -m apps.cohort_review`.
"""

import argparse
import csv
import json
import mimetypes
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import UUID

import cv2
from flask import Flask, Response, current_app, jsonify, request, send_file

from elephant_id.dataset import Dataset
from elephant_id.image.bgr import decode_image

from .audit import PairAudit
from .session import Candidate, ReviewSession

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_RESULTS = ROOT / "outputs/curation/archive/results.json"
ARCHIVE_CROPS = ARCHIVE_RESULTS.parent / "images"
DATASET_ROOT = ROOT / "dataset/elephants-alive/coded"
METADATA = ROOT / "dataset/elephants-alive/images.csv"
REVIEW_ROOT = ROOT / "outputs/curation/sighting-pair-review"
AUDIT_LONG_SIDE = 1600
AUDIT_BOX_RGB = (216, 168, 94)


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


def create_app(session: ReviewSession, audit: PairAudit, metadata_path: Path = METADATA) -> Flask:
    """Create the thin local Flask application around a review session and its audit."""
    app = Flask(__name__)
    with metadata_path.open(newline="") as handle:
        app.config["SIGHTING_DATES"] = {
            row["sighting_id"]: row["date"] for row in csv.DictReader(handle)
        }
    lock = threading.Lock()

    @app.get("/")
    def index() -> str:
        """Serve the side-at-a-time review workspace."""
        return _PAGE.replace("__MODE__", "review")

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

    @app.post("/api/forward")
    def forward() -> tuple[Response, int] | Response:
        """Navigate to the next review screen."""
        return _mutate(lock, session.forward, session)

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

    @app.get("/approved")
    def approved() -> str:
        """Serve the gallery of every approved sighting ear pair."""
        return _APPROVED_PAGE

    @app.get("/api/approved")
    def approved_pairs() -> Response:
        """Return approved pairs with sighting dates and crop routes."""
        dates = current_app.config["SIGHTING_DATES"]
        pairs = session.export()["approved_pairs"]
        for pair in pairs:
            pair["sighting_date"] = dates.get(pair["sighting_id"])
            pair["left_crop_path"] = f"/crop/{pair['left_crop_key']}"
            pair["right_crop_path"] = f"/crop/{pair['right_crop_key']}"
        return jsonify(pairs)

    def visible(crop_key: str) -> Candidate | None:
        """Return a crop on a review or replace screen, in an approved pair, or in a flagged pair."""
        if any(item.crop_key == crop_key for item in session.screen().candidates):
            return session.candidate(crop_key)
        with lock:
            if crop_key in audit.visible_crop_keys():
                return session.candidate(crop_key)
        return session.approved_candidate(crop_key)

    @app.get("/crop/<crop_key>")
    def crop(crop_key: str) -> Response:
        """Serve one frozen crop that is currently visible."""
        candidate = visible(crop_key)
        if candidate is None or not candidate.available:
            return Response("Crop unavailable", status=404)
        return send_file(candidate.crop_path)

    @app.get("/original/<crop_key>")
    def original(crop_key: str) -> Response:
        """Serve original sighting context for a visible crop."""
        candidate = visible(crop_key)
        if candidate is None or not candidate.original_path:
            return Response("Original unavailable", status=404)
        try:
            data = session.read_original(candidate)
        except FileNotFoundError:
            return Response("Original unavailable", status=404)
        media_type = mimetypes.guess_type(candidate.original_path)[0] or "application/octet-stream"
        return Response(data, mimetype=media_type)

    @app.get("/audit")
    def audit_page() -> str:
        """Serve the one-pair-at-a-time audit of approved pairs."""
        return _AUDIT_PAGE

    @app.get("/api/audit")
    def audit_screen() -> Response:
        """Return the current audit pair."""
        with lock:
            return jsonify(_web_audit(audit.audit_screen()))

    @app.post("/api/audit/move")
    def audit_move() -> tuple[Response, int] | Response:
        """Move to the previous or next approved pair."""
        return _respond(lock, lambda: _web_audit(audit.move(int(_json_body()["step"]))))

    @app.post("/api/audit/flag")
    def audit_flag() -> tuple[Response, int] | Response:
        """Flag or unflag the current approved pair."""
        return _respond(lock, lambda: _web_audit(audit.toggle_flag()))

    @app.get("/audit/photo/<crop_key>")
    def audit_photo(crop_key: str) -> Response:
        """Serve a visible crop's downscaled original photo with its crop box drawn."""
        candidate = visible(crop_key)
        if candidate is None:
            return Response("Photo unavailable", status=404)
        try:
            image = decode_image(session.read_original(candidate))
        except (FileNotFoundError, ValueError):
            return Response("Photo unavailable", status=404)
        scale = min(1.0, AUDIT_LONG_SIDE / max(image.shape[:2]))
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        x1, y1, x2, y2 = (round(value * scale) for value in candidate.crop_box)
        thickness = max(3, round(max(image.shape[:2]) / 250))
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), AUDIT_BOX_RGB[::-1], thickness)
        encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
        response = Response(encoded, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "private, max-age=86400"
        return response

    @app.get("/replace")
    def replace_page() -> str:
        """Serve the picker over the flagged-sighting replace queue."""
        return _PAGE.replace("__MODE__", "replace")

    @app.get("/replace/api/screen")
    def replace_screen() -> Response:
        """Return the current replace screen."""
        with lock:
            return jsonify(_web_screen(_replace_dict(audit.replace_screen())))

    @app.post("/replace/api/select")
    def replace_select() -> tuple[Response, int] | Response:
        """Choose a crop for the flagged sighting; the right ear commits the new pair."""
        return _respond(lock, lambda: _web_screen(_replace_dict(audit.replace_select(_json_body()["crop_key"]))))

    @app.post("/replace/api/reject")
    def replace_reject() -> tuple[Response, int] | Response:
        """Reject the flagged sighting."""
        return _respond(lock, lambda: _web_screen(_replace_dict(audit.replace_reject())))

    @app.post("/replace/api/back")
    def replace_back() -> tuple[Response, int] | Response:
        """Move to the previous replace screen."""
        return _respond(lock, lambda: _web_screen(_replace_dict(audit.replace_step(-1))))

    @app.post("/replace/api/forward")
    def replace_forward() -> tuple[Response, int] | Response:
        """Move to the next replace screen."""
        return _respond(lock, lambda: _web_screen(_replace_dict(audit.replace_step(1))))

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


def _respond(lock: threading.Lock, action: Any) -> tuple[Response, int] | Response:
    """Serialize a mutation that returns its own JSON payload and expose errors."""
    try:
        with lock:
            return jsonify(action())
    except (KeyError, TypeError, ValueError, OSError) as error:
        return jsonify({"error": str(error)}), 400


def _replace_dict(screen: dict[str, object]) -> dict[str, object]:
    """Serialize a replace screen's candidates like the picker's."""
    return {
        **screen,
        "candidates": [
            {**asdict(candidate), "photo_id": str(candidate.photo_id), "sighting_id": str(candidate.sighting_id)}
            for candidate in screen["candidates"]
        ],
    }


def _web_audit(screen: dict[str, object]) -> dict[str, object]:
    """Add the sighting date and photo routes to an audit screen."""
    pair = screen["pair"]
    if pair is not None:
        screen["pair"] = {
            **pair,
            "sighting_date": current_app.config["SIGHTING_DATES"].get(pair["sighting_id"]),
            "left_photo_path": f"/audit/photo/{pair['left_crop_key']}",
            "right_photo_path": f"/audit/photo/{pair['right_crop_key']}",
        }
    screen["prefetch"] = [f"/audit/photo/{key}" for key in screen["prefetch"]]
    return screen


def _web_screen(value: dict[str, object]) -> dict[str, object]:
    """Replace private filesystem paths with local HTTP routes."""
    value["sighting_date"] = current_app.config["SIGHTING_DATES"].get(value.get("sighting_id"))
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
    session = load_session()
    audit = PairAudit(session, REVIEW_ROOT / "audit.json")
    create_app(session, audit).run(host=args.host, port=args.port, debug=False)


_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alphaphant sighting pair review</title><style>
:root{color-scheme:dark;--ink:#f4f0e7;--muted:#a7afa9;--line:#38423d;--accent:#d8a85e;--panel:#18201c}
*{box-sizing:border-box}body{margin:0;background:#101512;color:var(--ink);font:16px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:4;display:flex;align-items:center;gap:16px;min-height:72px;padding:12px 22px;background:#101512ee;border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}
h1{font:600 19px/1.2 Georgia,serif;margin:0}.meta{color:var(--muted);flex:1}.progress{font-weight:700;color:var(--accent)}button,a.inspect,a.nav{border:1px solid #667269;border-radius:7px;background:#202a25;color:var(--ink);padding:9px 13px;text-decoration:none;cursor:pointer}button:hover,a.inspect:hover,a.nav:hover{border-color:var(--accent)}button:disabled{opacity:.45;cursor:wait}
#error{display:none;position:sticky;top:72px;z-index:5;background:#7c2929;padding:12px 22px}.score{padding:22px 0 8px;border-bottom:1px solid var(--line);font:600 17px Georgia,serif;color:var(--accent)}
main{max-width:2240px;margin:auto;padding:0 24px 90px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));column-gap:24px;align-items:start;position:relative}.empty{grid-column:1/-1}.score-dividers{position:absolute;inset:0;pointer-events:none;overflow:visible;z-index:1}.candidate{min-width:0;margin:18px 0 34px;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}.candidate.selected{border:2px solid var(--accent)}
.choose{position:relative;display:block;width:100%;padding:0;border:0;border-radius:0;background:#090c0a}.choose img{display:block;margin:auto;max-width:100%;width:auto;min-height:420px;max-height:78vh;object-fit:contain}.choose.unavailable{min-height:420px;color:var(--muted)}.number{position:absolute;bottom:12px;left:12px;min-width:36px;padding:3px 10px;border-radius:7px;background:#101512cc;color:var(--ink);font:600 19.5px/1.4 system-ui,sans-serif;text-align:center;pointer-events:none}
.empty{max-width:680px;margin:18vh auto;text-align:center}.empty h2{font:600 34px Georgia,serif}
@media(max-width:650px){header{flex-wrap:wrap}.meta{order:3;flex-basis:100%}main{padding-inline:10px;column-gap:24px}.choose img,.choose.unavailable{min-height:300px}}
</style></head><body><header><button id="back">← Back</button><button id="forward">Forward →</button><div><h1 id="title">Loading review…</h1><div class="meta" id="meta"></div></div><div class="meta"></div><div class="progress" id="progress"></div><button id="reject">Reject sighting</button><a class="nav" href="/approved">Approved pairs</a><a class="nav" href="/audit">Audit</a><a class="nav review-only" id="export" href="/api/export" target="_blank" download>Backup</a><button class="review-only" id="restore">Restore</button><input id="restore-file" type="file" accept="application/json" hidden></header><div id="error"></div><main id="content"></main>
<script>
'use strict';const $=id=>document.getElementById(id),MODE='__MODE__',API=MODE==='replace'?'/replace/api':'/api';let busy=false,scrollTimer,current=null;if(MODE==='replace'){document.title='Alphaphant flagged pair replacement';document.querySelectorAll('.review-only').forEach(x=>x.hidden=true)}
async function call(path,body={}){if(busy)return;clearTimeout(scrollTimer);busy=true;document.querySelectorAll('button').forEach(x=>x.disabled=true);$('error').style.display='none';try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),data=await r.json();if(!r.ok)throw Error(data.error||'Save failed');render(data)}catch(e){$('error').textContent=`Not saved: ${e.message}`;$('error').style.display='block'}finally{busy=false;document.querySelectorAll('button:not(.choose),button.choose.loaded').forEach(x=>x.disabled=!!x.dataset.off)}}
function render(s){current=s;$('forward').dataset.off=s.can_forward?'':'1';$('forward').disabled=!s.can_forward;const replacing=MODE==='replace';if(s.status==='complete'){$('title').textContent=replacing?'Replace queue complete':'Review queue complete';$('meta').textContent=replacing?'Every flagged sighting is re-picked or rejected.':'All active score 10, 9, and 8 sightings are decided.';$('progress').textContent=replacing?`Resolved ${s.progress.resolved} / ${s.progress.flagged}`:`Elephants ${s.progress.elephants_complete} / ${s.progress.elephants_total} · Sightings ${s.progress.sightings_processed} / ${s.progress.sightings_target}`;$('reject').hidden=true;$('content').innerHTML='<section class="empty"><h2>Queue complete</h2><p>Your accepted pairs are saved. Back remains available for revisions.</p></section>';return}$('reject').hidden=false;$('title').textContent=`${s.elephant} · ${s.sighting_date||"Date unavailable"} · ${s.side} ear`;const action=`${s.candidates.length} assessed candidates · click an image to ${s.side==='left'?'continue':'save pair'}`;const p=s.progress;if(replacing){$('meta').textContent=[s.reason&&`Flagged: ${s.reason}`,s.resolution&&`already ${s.resolution}`,action].filter(Boolean).join(' · ');$('progress').textContent=`Resolved ${p.resolved} / ${p.flagged}`}else{$('meta').textContent=`Proposed tier ${s.tier} · ${action}`;$('progress').textContent=`Elephant ${p.elephant_processed} / ${p.elephant_usable} · ${p.elephant_accepted} / 10 accepted · Elephants ${p.elephants_complete} / ${p.elephants_total} · Sightings ${p.sightings_processed} / ${p.sightings_target}`}
current=s;let html='';for(const[i,c]of s.candidates.entries()){html+=`<article data-score="${c.score}" class="candidate ${c.crop_key===s.selected_crop_key?'selected':''}"><button class="choose ${c.available?'':'unavailable'}" data-key="${c.crop_key}" disabled><span class="number">${i+1}</span>${c.available?`<img src="${c.crop_path}" alt="${s.side} ear candidate, score ${c.score}" loading="lazy">`:'Image unavailable'}</button></article>`}$('content').innerHTML=html+'<svg class="score-dividers" aria-hidden="true"></svg>';drawDividers();document.querySelectorAll('.choose').forEach(x=>{const img=x.querySelector('img');if(img){img.onload=()=>{x.classList.add('loaded');if(!busy)x.disabled=false};img.onerror=()=>{x.classList.add('unavailable');x.textContent='Image unavailable'}}x.onclick=()=>call(API+'/select',{crop_key:x.dataset.key})});requestAnimationFrame(()=>scrollTo(0,s.scroll_y||0))}
function drawDividers(){const grid=$('content'),svg=grid.querySelector('.score-dividers');if(!svg)return;const bounds=grid.getBoundingClientRect(),cards=[...grid.querySelectorAll('.candidate')],boxes=cards.map(c=>{const r=c.getBoundingClientRect();return {left:r.left-bounds.left,right:r.right-bounds.left,top:r.top-bounds.top,bottom:r.bottom-bounds.top}});svg.setAttribute('width',bounds.width);svg.setAttribute('height',grid.scrollHeight);let paths='';for(let i=0;i<cards.length;i++){if(i&&cards[i].dataset.score===cards[i-1].dataset.score)continue;const b=boxes[i],left=boxes[0].left,right=boxes.length>1?boxes[1].right:bounds.width-left;let d;if(i%2===0){const followsStep=i>=2&&cards[i-2].dataset.score!==cards[i-1].dataset.score,mid=boxes.length>1?(boxes[0].right+boxes[1].left)/2:left;d=`M ${followsStep?mid:left} ${Math.max(4,b.top-26)+(followsStep?6:0)} H ${right}`}else{const a=boxes[i-1],mid=(a.right+b.left)/2,top=Math.max(4,Math.min(a.top,b.top)-26),anotherBoundary=i+1<cards.length&&cards[i+1].dataset.score!==cards[i].dataset.score,bottom=Math.max(a.bottom,b.bottom)+26-(anotherBoundary?6:0);d=`M ${right} ${top} H ${mid} V ${bottom} H ${left}`}paths+=`<path d="${d}" fill="none" stroke="#d8a85e" stroke-width="2" stroke-linejoin="round"/>`}svg.innerHTML=paths}new ResizeObserver(drawDividers).observe($('content'));
$('back').onclick=()=>call(API+'/back');$('forward').onclick=()=>call(API+'/forward');$('reject').onclick=()=>call(API+'/reject');$('restore').onclick=()=>$('restore-file').click();$('restore-file').onchange=async e=>{try{call('/api/restore',JSON.parse(await e.target.files[0].text()))}catch(err){$('error').textContent=`Restore failed: ${err.message}`;$('error').style.display='block'}};document.addEventListener('keydown',e=>{if(e.repeat||e.altKey||e.ctrlKey||e.metaKey||['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowLeft'){e.preventDefault();call(API+'/back')}else if(e.key==='ArrowRight'&&current?.can_forward){e.preventDefault();call(API+'/forward')}});addEventListener('scroll',()=>{if(MODE!=='review'||!current||current.status!=='review')return;clearTimeout(scrollTimer);const location={sighting_id:current.sighting_id,side:current.side,scroll_y:scrollY};scrollTimer=setTimeout(()=>fetch('/api/scroll',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(location)}),250)});fetch(API+'/screen').then(r=>r.json()).then(render).catch(e=>{$('error').textContent=e.message;$('error').style.display='block'});
</script></body></html>'''


_APPROVED_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alphaphant approved pairs</title><style>
:root{color-scheme:dark;--ink:#f4f0e7;--muted:#a7afa9;--line:#38423d;--accent:#d8a85e;--panel:#18201c}
*{box-sizing:border-box}body{margin:0;background:#101512;color:var(--ink);font:16px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:4;display:flex;align-items:center;gap:16px;min-height:72px;padding:12px 22px;background:#101512ee;border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}
h1{font:600 19px/1.2 Georgia,serif;margin:0}.meta{color:var(--muted);flex:1}a.nav{border:1px solid #667269;border-radius:7px;background:#202a25;color:var(--ink);padding:9px 13px;text-decoration:none}a.nav:hover{border-color:var(--accent)}
main{max-width:1600px;margin:auto;padding:0 24px 90px}h2{font:600 22px Georgia,serif;color:var(--accent);margin:34px 0 4px;padding-bottom:8px;border-bottom:1px solid var(--line)}h2 span{color:var(--muted);font:15px system-ui,sans-serif;margin-left:10px}
.pair{margin:18px 0;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}.pair-head{display:flex;flex-wrap:wrap;gap:6px 18px;padding:10px 14px;color:var(--muted);font-size:14px}.pair-head strong{color:var(--ink)}
.ears{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:2px;background:var(--line)}figure{margin:0;background:#090c0a}figure img{display:block;margin:auto;max-width:100%;height:340px;object-fit:contain}figcaption{padding:6px 12px;color:var(--muted);font-size:13px;background:var(--panel)}
#error{display:none;background:#7c2929;padding:12px 22px}
@media(max-width:650px){main{padding-inline:10px}figure img{height:200px}}
</style></head><body><header><a class="nav" href="/">← Review</a><div><h1>Approved sighting ear pairs</h1><div class="meta" id="meta">Loading…</div></div></header><div id="error"></div><main id="content"></main>
<script>
'use strict';const $=id=>document.getElementById(id),esc=v=>String(v).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function ear(p,side){return `<figure><a href="${p[side+'_crop_path']}" target="_blank"><img src="${p[side+'_crop_path']}" alt="${side} ear" loading="lazy"></a><figcaption>${side[0].toUpperCase()+side.slice(1)} · score ${p[side+'_score']} · photo ${p[side+'_photo_id']}</figcaption></figure>`}
fetch('/api/approved').then(r=>r.json()).then(pairs=>{const groups=new Map;for(const p of pairs){if(!groups.has(p.elephant))groups.set(p.elephant,[]);groups.get(p.elephant).push(p)}$('meta').textContent=`${pairs.length} pairs · ${groups.size} elephants`;let html='';for(const[name,items]of groups){items.sort((a,b)=>String(a.sighting_date).localeCompare(String(b.sighting_date)));html+=`<h2>${esc(name)}<span>${items.length} pair${items.length===1?'':'s'}</span></h2>`;for(const p of items)html+=`<article class="pair"><div class="pair-head"><strong>${esc(p.sighting_date||'Date unavailable')}</strong><span>Sighting ${p.sighting_id}</span></div><div class="ears">${ear(p,'left')}${ear(p,'right')}</div></article>`}$('content').innerHTML=html||'<p>No approved pairs yet.</p>'}).catch(e=>{$('error').textContent=e.message;$('error').style.display='block'});
</script></body></html>'''


_AUDIT_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alphaphant pair audit</title><style>
:root{color-scheme:dark;--ink:#f4f0e7;--muted:#a7afa9;--line:#38423d;--accent:#d8a85e;--panel:#18201c;--flag:#e0604f}
*{box-sizing:border-box}body{margin:0;background:#101512;color:var(--ink);font:16px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:4;display:flex;flex-wrap:wrap;align-items:center;gap:12px 16px;min-height:72px;padding:12px 22px;background:#101512ee;border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}
h1{font:600 19px/1.2 Georgia,serif;margin:0}.meta{color:var(--muted);font-size:14px}.grow{flex:1}.progress{font-weight:700;color:var(--accent)}
button,a.nav{border:1px solid #667269;border-radius:7px;background:#202a25;color:var(--ink);padding:9px 13px;text-decoration:none;cursor:pointer;font:inherit}button:hover,a.nav:hover{border-color:var(--accent)}
.badge{display:none;padding:4px 10px;border-radius:7px;font-weight:700;font-size:14px}.badge.flagged{display:inline-block;background:var(--flag);color:#fff}.badge.repicked,.badge.rejected{display:inline-block;background:#2f5a3d}
#error{display:none;background:#7c2929;padding:12px 22px}
main{max-width:2240px;margin:auto;padding:18px 24px 60px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}main.flagged figure{border-color:var(--flag);box-shadow:0 0 0 2px var(--flag)}
figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}figure img{display:block;margin:auto;max-width:100%;max-height:calc(100vh - 190px);min-height:300px;object-fit:contain;background:#090c0a}
figcaption{display:flex;flex-wrap:wrap;gap:4px 14px;padding:9px 14px;color:var(--muted);font-size:14px}figcaption strong{color:var(--ink);text-transform:capitalize}
.keys{color:var(--muted);font-size:13px}kbd{border:1px solid #667269;border-radius:4px;padding:0 5px;font:12px ui-monospace,monospace}
.empty{grid-column:1/-1;text-align:center;margin:18vh auto}
@media(max-width:650px){header{padding-inline:12px}main{grid-template-columns:1fr;padding-inline:12px}figure img{min-height:200px;max-height:60vh}}
</style></head><body><header><button id="prev">← Previous</button><button id="next">Next →</button><div class="grow"><h1 id="title">Loading audit…</h1><div class="meta" id="meta"></div></div><span class="badge" id="badge"></span><button id="flag">Flag</button><span class="progress" id="progress"></span><a class="nav" href="/replace" id="replace">Replace flagged</a><a class="nav" href="/">Picker</a><span class="keys"><kbd>←</kbd> <kbd>→</kbd> move · <kbd>Space</kbd> flag</span></header><div id="error"></div><main id="content"></main>
<script>
'use strict';const $=id=>document.getElementById(id),esc=v=>String(v).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));let busy=false,current=null;const cache=new Map;
function prefetch(urls){for(const url of urls)if(!cache.has(url)){const img=new Image;img.src=url;cache.set(url,img);if(cache.size>40)cache.delete(cache.keys().next().value)}}
function ear(p,side){return `<figure><img src="${p[side+'_photo_path']}" alt="${side} ear crop in its full photo"><figcaption><strong>Declared ${side} ear</strong><span>score ${p[side+'_score']}</span><span>photo ${p[side+'_photo_id'].slice(0,8)}</span><span>box ${p[side+'_crop_box'].join(', ')}</span></figcaption></figure>`}
function render(s){current=s;const p=s.pair;$('progress').textContent=`${s.flagged} flagged · ${s.resolved} resolved`;$('replace').textContent=`Replace flagged (${s.flagged})`;if(!p){$('title').textContent='No approved pairs';$('meta').textContent='';$('content').innerHTML='<p class="empty">Approve pairs in the picker first.</p>';return}
$('title').textContent=`${p.elephant} · ${p.sighting_date||'Date unavailable'}`;$('meta').textContent=`Pair ${s.index+1} / ${s.total} · sighting ${p.sighting_id}${s.reason?` · ${s.reason}`:''}`;const badge=$('badge');badge.className=`badge ${s.status||''}`;badge.textContent=s.status==='flagged'?'Flagged':s.status?`Flag ${s.status}`:'';$('flag').textContent=s.status==='flagged'?'Unflag':'Flag';$('prev').disabled=s.index===0;$('next').disabled=s.index+1>=s.total;
const main=$('content');main.className=s.status==='flagged'?'flagged':'';main.innerHTML=ear(p,'left')+ear(p,'right');prefetch(s.prefetch)}
async function post(path,body={}){if(busy)return;busy=true;$('error').style.display='none';try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),data=await r.json();if(!r.ok)throw Error(data.error||'Save failed');render(data)}catch(e){$('error').textContent=`Not saved: ${e.message}`;$('error').style.display='block'}finally{busy=false}}
const move=step=>post('/api/audit/move',{step}),flag=()=>post('/api/audit/flag');
$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('flag').onclick=flag;
document.addEventListener('keydown',e=>{if(e.altKey||e.ctrlKey||e.metaKey)return;if(e.key==='ArrowLeft'){e.preventDefault();move(-1)}else if(e.key==='ArrowRight'){e.preventDefault();move(1)}else if(e.key===' '){e.preventDefault();document.activeElement?.blur();if(!e.repeat)flag()}});
fetch('/api/audit').then(r=>r.json()).then(render).catch(e=>{$('error').textContent=e.message;$('error').style.display='block'});
</script></body></html>'''


if __name__ == "__main__":
    main()

"use strict";

const elephantList = document.getElementById("elephant-list");
const elephantPanel = document.getElementById("elephant-panel");
const scanStatus = document.getElementById("scan-status");
const reviewProgress = document.getElementById("review-progress");
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");
const lightboxCaption = document.getElementById("lightbox-caption");
const lightboxClose = document.getElementById("lightbox-close");

let selectedIdentity = null;

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function postJSON(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data && data.error ? data.error : response.statusText;
    throw new Error(detail);
  }
  return data;
}

function cropUrl(identity, sightingDate, side, candidateId) {
  const params = new URLSearchParams({
    identity,
    sightingDate,
    side,
    candidateId,
  });
  return `/api/crop?${params.toString()}`;
}

/* --- lightbox ------------------------------------------------------------ */
const hasLightbox = Boolean(lightbox && lightboxImg && lightboxCaption);

function openLightbox(url, caption) {
  if (!hasLightbox) {
    return;
  }
  lightboxImg.src = url;
  lightboxCaption.innerHTML = caption;
  lightbox.classList.add("open");
  lightbox.setAttribute("aria-hidden", "false");
}

function closeLightbox() {
  if (!hasLightbox) {
    return;
  }
  lightbox.classList.remove("open");
  lightbox.setAttribute("aria-hidden", "true");
  lightboxImg.removeAttribute("src");
}

if (hasLightbox) {
  if (lightboxClose) {
    lightboxClose.addEventListener("click", closeLightbox);
  }
  lightbox.addEventListener("click", (event) => {
    if (event.target === lightbox) {
      closeLightbox();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && lightbox.classList.contains("open")) {
      closeLightbox();
    }
  });
}

/* --- sidebar ------------------------------------------------------------- */
function renderScan(scan) {
  if (scan.running) {
    scanStatus.textContent =
      `Scanning ${scan.scanned}/${scan.total} ` +
      `(${scan.eligible} eligible)` +
      (scan.current ? ` — ${scan.current}` : "");
  } else {
    scanStatus.textContent = `Scan complete — ${scan.eligible} eligible elephants`;
  }
}

function renderReviewProgress(view) {
  const total = view.elephants.length;
  reviewProgress.textContent = total
    ? `${view.doneCount} of ${total} elephants done`
    : "";
}

function elephantItem(elephant) {
  const item = document.createElement("li");
  item.className = "elephant-item";
  item.dataset.identity = elephant.identity;
  if (elephant.identity === selectedIdentity) {
    item.classList.add("selected");
  }
  if (elephant.done) {
    item.classList.add("done");
  }
  const name = document.createElement("span");
  name.className = "name";
  name.textContent = elephant.identity;
  item.appendChild(name);
  const badge = document.createElement("span");
  badge.className = "badge";
  // Show complete-out-of-selected so partially-picked sightings are visible;
  // fall back to the minimum target until at least that many are selected.
  const denominator = Math.max(elephant.selectedCount, elephant.minSightings);
  badge.textContent = elephant.done
    ? `✓ ${elephant.completeCount}`
    : `${elephant.completeCount}/${denominator}`;
  item.appendChild(badge);
  item.addEventListener("click", () => openElephant(elephant.identity));
  return item;
}

function divider(text) {
  const li = document.createElement("li");
  li.className = "list-divider";
  li.textContent = text;
  return li;
}

function renderElephants(elephants) {
  elephantList.innerHTML = "";
  // Group done elephants to the top ("in done order"), each group keeping the
  // dataset's own ordering. Dividers only appear once something is done.
  const done = elephants.filter((elephant) => elephant.done);
  const remaining = elephants.filter((elephant) => !elephant.done);
  if (done.length) {
    elephantList.appendChild(divider(`Done (${done.length})`));
    for (const elephant of done) {
      elephantList.appendChild(elephantItem(elephant));
    }
    elephantList.appendChild(divider(`Remaining (${remaining.length})`));
  }
  for (const elephant of remaining) {
    elephantList.appendChild(elephantItem(elephant));
  }
}

async function refreshElephants() {
  const view = await getJSON("/api/elephants");
  renderScan(view.scan);
  renderReviewProgress(view);
  renderElephants(view.elephants);
  if (view.scan.running) {
    setTimeout(refreshElephants, 2000);
  }
}

/* --- candidates ---------------------------------------------------------- */
function renderCandidate(sighting, side, candidate, rank) {
  const figure = document.createElement("figure");
  figure.className = "candidate";
  if (rank === 1) {
    figure.classList.add("top");
  }
  if (candidate.picked) {
    figure.classList.add("picked");
  }

  const frame = document.createElement("div");
  frame.className = "frame";
  const url = cropUrl(
    selectedIdentity,
    sighting.sightingDate,
    side,
    candidate.candidateId,
  );
  const img = document.createElement("img");
  img.loading = "lazy";
  img.src = url;
  img.alt = `${side} ear candidate for ${sighting.sightingDate}`;
  frame.appendChild(img);

  const rankBadge = document.createElement("span");
  rankBadge.className = "rank-badge";
  rankBadge.textContent = `#${rank}`;
  frame.appendChild(rankBadge);

  const zoom = document.createElement("button");
  zoom.className = "zoom-btn";
  zoom.type = "button";
  zoom.title = "Enlarge to inspect detail";
  zoom.setAttribute("aria-label", "Enlarge candidate");
  zoom.textContent = "⤢";
  zoom.addEventListener("click", (event) => {
    event.stopPropagation();
    openLightbox(
      url,
      `<strong>${candidate.photoIdentifier}</strong> · ${side} ear · ` +
        `score ${candidate.quality.toFixed(3)}`,
    );
  });
  frame.appendChild(zoom);

  if (candidate.picked) {
    const check = document.createElement("span");
    check.className = "check-badge";
    check.innerHTML =
      '<span class="glyph keep">✓</span><span class="glyph drop">✕</span>';
    frame.appendChild(check);
  }
  figure.appendChild(frame);

  const meta = document.createElement("figcaption");
  meta.className = "candidate-meta";
  const score = document.createElement("span");
  score.className = "quality-score";
  score.textContent = candidate.quality.toFixed(3);
  meta.appendChild(score);
  if (candidate.picked) {
    const label = document.createElement("span");
    label.className = "picked-label";
    label.innerHTML =
      '<span class="on">Picked</span><span class="off">Remove</span>';
    meta.appendChild(label);
  }
  figure.appendChild(meta);

  figure.title = candidate.picked ? "Click to un-select" : "Click to select";
  figure.addEventListener("click", () => {
    if (candidate.picked) {
      unpickCandidate(sighting.sightingDate, side);
    } else {
      pickCandidate(sighting.sightingDate, side, candidate.candidateId);
    }
  });
  return figure;
}

function renderSideBlock(sighting, side) {
  const block = document.createElement("div");
  block.className = "side-block";
  const heading = document.createElement("h4");
  const picked = sighting[side].some((candidate) => candidate.picked);
  heading.textContent =
    (side === "left" ? "Left ear" : "Right ear") + (picked ? " ✓" : "");
  block.appendChild(heading);
  const grid = document.createElement("div");
  grid.className = "candidate-grid";
  grid.dataset.side = side;
  sighting[side].forEach((candidate, index) => {
    grid.appendChild(renderCandidate(sighting, side, candidate, index + 1));
  });
  block.appendChild(grid);
  return block;
}

function renderSighting(sighting) {
  const card = document.createElement("section");
  card.className = "sighting-card";
  if (sighting.complete) {
    card.classList.add("complete");
  } else if (sighting.selected) {
    card.classList.add("partial");
  }
  card.dataset.sightingDate = sighting.sightingDate;

  const head = document.createElement("div");
  head.className = "sighting-head";
  const title = document.createElement("h3");
  title.textContent = sighting.sightingDate;
  head.appendChild(title);
  if (sighting.complete) {
    const pill = document.createElement("span");
    pill.className = "status-pill complete";
    pill.textContent = "Complete";
    head.appendChild(pill);
  } else if (sighting.selected) {
    const pill = document.createElement("span");
    pill.className = "status-pill partial";
    pill.textContent = "Needs both ears";
    head.appendChild(pill);
  }
  if (sighting.segmentationOverlap) {
    const pill = document.createElement("span");
    pill.className = "status-pill overlap";
    pill.title =
      "A photo from this sighting is already in the segmentation annotation " +
      "batch. Still selectable — this is for awareness only.";
    pill.textContent = "In segmentation batch";
    head.appendChild(pill);
  }
  card.appendChild(head);

  const columns = document.createElement("div");
  columns.className = "side-columns";
  columns.appendChild(renderSideBlock(sighting, "left"));
  columns.appendChild(renderSideBlock(sighting, "right"));
  card.appendChild(columns);
  return card;
}

/* --- selection counter --------------------------------------------------- */
function selectionText(selection) {
  const range = `${selection.minSightings}–${selection.maxSightings}`;
  let text =
    `Selected ${selection.selectedCount} of ${range} sightings ` +
    `(${selection.completeCount} complete)`;
  if (selection.done) {
    text += " — done ✓";
  }
  return text;
}

function updateSelection(selection) {
  const counter = document.getElementById("selection-counter");
  if (counter) {
    counter.classList.toggle("done", selection.done);
    counter.textContent = selectionText(selection);
  }
}

function renderElephantPanel(view) {
  elephantPanel.innerHTML = "";

  const header = document.createElement("div");
  header.className = "panel-header";
  const heading = document.createElement("h2");
  heading.textContent = view.identity;
  const subtitle = document.createElement("span");
  subtitle.className = "subtitle";
  subtitle.textContent = `${view.qualifyingCount} sightings to review`;
  heading.appendChild(document.createTextNode(" "));
  heading.appendChild(subtitle);
  header.appendChild(heading);

  const counter = document.createElement("div");
  counter.id = "selection-counter";
  counter.className = "selection-counter";
  counter.classList.toggle("done", view.selection.done);
  counter.textContent = selectionText(view.selection);
  header.appendChild(counter);
  elephantPanel.appendChild(header);

  const sightings = document.createElement("div");
  sightings.className = "sightings";
  for (const sighting of view.sightings) {
    sightings.appendChild(renderSighting(sighting));
  }
  elephantPanel.appendChild(sightings);
}

async function openElephant(identity) {
  selectedIdentity = identity;
  for (const item of elephantList.children) {
    if (item.dataset && item.dataset.identity) {
      item.classList.toggle("selected", item.dataset.identity === identity);
    }
  }
  elephantPanel.innerHTML = "<p class='placeholder'>Loading&hellip;</p>";
  const view = await getJSON(`/api/elephant/${encodeURIComponent(identity)}`);
  renderElephantPanel(view);
}

function replaceSighting(payload) {
  const sightings = elephantPanel.querySelector(".sightings");
  const card = elephantPanel.querySelector(
    `.sighting-card[data-sighting-date="${payload.sightingDate}"]`,
  );
  if (card && sightings) {
    card.replaceWith(renderSighting(payload));
  }
}

async function pickCandidate(sightingDate, side, candidateId) {
  try {
    const result = await postJSON("/api/pick", {
      identity: selectedIdentity,
      sightingDate,
      side,
      candidateId,
    });
    replaceSighting(result.sighting);
    updateSelection(result.selection);
    refreshElephants().catch(() => {});
  } catch (error) {
    scanStatus.textContent = `Pick rejected: ${error.message}`;
  }
}

async function unpickCandidate(sightingDate, side) {
  try {
    const result = await postJSON("/api/unpick", {
      identity: selectedIdentity,
      sightingDate,
      side,
    });
    replaceSighting(result.sighting);
    updateSelection(result.selection);
    refreshElephants().catch(() => {});
  } catch (error) {
    scanStatus.textContent = `Un-select rejected: ${error.message}`;
  }
}

refreshElephants().catch((error) => {
  scanStatus.textContent = `Error: ${error.message}`;
});

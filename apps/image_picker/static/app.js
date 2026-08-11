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
// Ordered elephant identities as shown in the sidebar (done first, then
// remaining) — the sequence the left/right arrow keys step through.
let orderedIdentities = [];
// The elephant view currently rendered in the panel, kept so the "selected
// only" filter and in-place card updates can re-derive what to show.
let currentView = null;
// When true, the panel shows only sightings that already have a pick.
let showSelectedOnly = false;

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
  orderedIdentities = [...done, ...remaining].map((elephant) => elephant.identity);
}

/* --- keyboard navigation ------------------------------------------------- */
function moveElephant(delta) {
  if (!orderedIdentities.length) {
    return;
  }
  const current = orderedIdentities.indexOf(selectedIdentity);
  if (current === -1) {
    openElephant(orderedIdentities[0]);
    return;
  }
  const next = current + delta;
  if (next < 0 || next >= orderedIdentities.length) {
    return;
  }
  const identity = orderedIdentities[next];
  openElephant(identity);
  const item = elephantList.querySelector(
    `.elephant-item[data-identity="${CSS.escape(identity)}"]`,
  );
  if (item) {
    item.scrollIntoView({ block: "nearest" });
  }
}

document.addEventListener("keydown", (event) => {
  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
    return;
  }
  if (event.metaKey || event.ctrlKey || event.altKey) {
    return;
  }
  // Leave the arrows alone for the lightbox and any focused text field.
  if (lightbox && lightbox.classList.contains("open")) {
    return;
  }
  const active = document.activeElement;
  if (active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA")) {
    return;
  }
  event.preventDefault();
  moveElephant(event.key === "ArrowRight" ? 1 : -1);
});

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
  // In "selected only" mode show just the chosen crop for each side, enlarged,
  // rather than the full ranked candidate list.
  grid.classList.toggle("picked-only", showSelectedOnly);
  let shown = 0;
  sighting[side].forEach((candidate, index) => {
    if (showSelectedOnly && !candidate.picked) {
      return;
    }
    grid.appendChild(renderCandidate(sighting, side, candidate, index + 1));
    shown += 1;
  });
  if (showSelectedOnly && shown === 0) {
    const empty = document.createElement("p");
    empty.className = "side-empty";
    empty.textContent = "Not picked yet";
    grid.appendChild(empty);
  }
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

function sightingsToShow(view) {
  return showSelectedOnly
    ? view.sightings.filter((sighting) => sighting.selected)
    : view.sightings;
}

function setShowSelectedOnly(value) {
  if (showSelectedOnly === value) {
    return;
  }
  showSelectedOnly = value;
  if (currentView) {
    renderElephantPanel(currentView);
  }
}

function viewToggle(view) {
  const selectedCount = view.sightings.filter((s) => s.selected).length;
  const toggle = document.createElement("div");
  toggle.className = "view-toggle";
  toggle.setAttribute("role", "group");
  toggle.setAttribute("aria-label", "Filter sightings");
  const options = [
    ["all", "All", false],
    ["selected", `Selected (${selectedCount})`, true],
  ];
  for (const [, label, selectedOnly] of options) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "view-toggle-btn";
    button.textContent = label;
    const active = selectedOnly === showSelectedOnly;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
    button.addEventListener("click", () => setShowSelectedOnly(selectedOnly));
    toggle.appendChild(button);
  }
  return toggle;
}

function renderSightingsList(view) {
  const sightings = document.createElement("div");
  sightings.className = "sightings";
  const toShow = sightingsToShow(view);
  if (!toShow.length) {
    const empty = document.createElement("p");
    empty.className = "placeholder";
    empty.textContent = showSelectedOnly
      ? "No sightings selected yet — switch to All to start picking."
      : "No qualifying sightings.";
    sightings.appendChild(empty);
  } else {
    for (const sighting of toShow) {
      sightings.appendChild(renderSighting(sighting));
    }
  }
  return sightings;
}

function renderElephantPanel(view) {
  currentView = view;
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

  const controls = document.createElement("div");
  controls.className = "panel-header-controls";
  const counter = document.createElement("div");
  counter.id = "selection-counter";
  counter.className = "selection-counter";
  counter.classList.toggle("done", view.selection.done);
  counter.textContent = selectionText(view.selection);
  controls.appendChild(counter);
  controls.appendChild(viewToggle(view));
  header.appendChild(controls);
  elephantPanel.appendChild(header);

  elephantPanel.appendChild(renderSightingsList(view));
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

function refreshViewToggle() {
  const existing = elephantPanel.querySelector(".view-toggle");
  if (existing && currentView) {
    existing.replaceWith(viewToggle(currentView));
  }
}

function replaceSighting(payload) {
  if (currentView) {
    const index = currentView.sightings.findIndex(
      (sighting) => sighting.sightingDate === payload.sightingDate,
    );
    if (index !== -1) {
      currentView.sightings[index] = payload;
    }
  }
  const sightings = elephantPanel.querySelector(".sightings");
  const card = elephantPanel.querySelector(
    `.sighting-card[data-sighting-date="${payload.sightingDate}"]`,
  );
  if (showSelectedOnly && !payload.selected) {
    // The sighting lost its last pick; drop it from the selected-only view,
    // falling back to the empty-state message if nothing selected remains.
    if (card) {
      card.remove();
    }
    if (sightings && currentView && !sightings.querySelector(".sighting-card")) {
      sightings.replaceWith(renderSightingsList(currentView));
    }
  } else if (card) {
    card.replaceWith(renderSighting(payload));
  }
  refreshViewToggle();
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

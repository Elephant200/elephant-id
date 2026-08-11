"use strict";

const elephantList = document.getElementById("elephant-list");
const elephantPanel = document.getElementById("elephant-panel");
const scanStatus = document.getElementById("scan-status");
const reviewProgress = document.getElementById("review-progress");

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

function renderElephants(elephants) {
  elephantList.innerHTML = "";
  for (const elephant of elephants) {
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
    elephantList.appendChild(item);
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

function renderCandidate(sighting, side, candidate) {
  const figure = document.createElement("figure");
  figure.className = "candidate";
  if (candidate.picked) {
    figure.classList.add("picked");
  }
  const img = document.createElement("img");
  img.loading = "lazy";
  img.src = cropUrl(
    selectedIdentity,
    sighting.sightingDate,
    side,
    candidate.candidateId,
  );
  const caption = document.createElement("figcaption");
  caption.textContent = candidate.quality.toFixed(3);
  figure.appendChild(img);
  figure.appendChild(caption);
  figure.addEventListener("click", () =>
    pickCandidate(sighting.sightingDate, side, candidate.candidateId),
  );
  return figure;
}

function renderSideColumn(sighting, side) {
  const column = document.createElement("div");
  column.className = "side-column";
  const heading = document.createElement("h4");
  heading.textContent = side === "left" ? "Left ear" : "Right ear";
  column.appendChild(heading);
  const strip = document.createElement("div");
  strip.className = "candidate-strip";
  strip.dataset.side = side;
  for (const candidate of sighting[side]) {
    strip.appendChild(renderCandidate(sighting, side, candidate));
  }
  column.appendChild(strip);
  return column;
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
  const header = document.createElement("h3");
  header.textContent = sighting.sightingDate;
  if (sighting.complete) {
    header.textContent += " ✓";
  } else if (sighting.selected) {
    header.textContent += " (needs both ears)";
  }
  card.appendChild(header);
  const columns = document.createElement("div");
  columns.className = "side-columns";
  columns.appendChild(renderSideColumn(sighting, "left"));
  columns.appendChild(renderSideColumn(sighting, "right"));
  card.appendChild(columns);
  return card;
}

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

function renderSelectionCounter(selection) {
  const counter = document.createElement("div");
  counter.id = "selection-counter";
  counter.className = "selection-counter";
  counter.classList.toggle("done", selection.done);
  counter.textContent = selectionText(selection);
  return counter;
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
  const heading = document.createElement("h2");
  heading.textContent =
    `${view.identity} — ${view.qualifyingCount} qualifying sightings`;
  elephantPanel.appendChild(heading);
  elephantPanel.appendChild(renderSelectionCounter(view.selection));
  for (const sighting of view.sightings) {
    elephantPanel.appendChild(renderSighting(sighting));
  }
}

async function openElephant(identity) {
  selectedIdentity = identity;
  for (const item of elephantList.children) {
    item.classList.toggle("selected", item.dataset.identity === identity);
  }
  elephantPanel.innerHTML = "<p class='placeholder'>Loading&hellip;</p>";
  const view = await getJSON(`/api/elephant/${encodeURIComponent(identity)}`);
  renderElephantPanel(view);
}

function replaceSighting(payload) {
  const card = elephantPanel.querySelector(
    `.sighting-card[data-sighting-date="${payload.sightingDate}"]`,
  );
  if (card) {
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

refreshElephants().catch((error) => {
  scanStatus.textContent = `Error: ${error.message}`;
});

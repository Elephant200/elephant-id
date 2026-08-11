"use strict";

const elephantList = document.getElementById("elephant-list");
const elephantPanel = document.getElementById("elephant-panel");
const scanStatus = document.getElementById("scan-status");

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
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
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

function renderElephants(elephants) {
  elephantList.innerHTML = "";
  for (const elephant of elephants) {
    const item = document.createElement("li");
    item.className = "elephant-item";
    item.dataset.identity = elephant.identity;
    if (elephant.identity === selectedIdentity) {
      item.classList.add("selected");
    }
    const name = document.createElement("span");
    name.textContent = elephant.identity;
    item.appendChild(name);
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = `${elephant.pickedCount} picked`;
    item.appendChild(badge);
    item.addEventListener("click", () => openElephant(elephant.identity));
    elephantList.appendChild(item);
  }
}

async function refreshElephants() {
  const view = await getJSON("/api/elephants");
  renderScan(view.scan);
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
  card.dataset.sightingDate = sighting.sightingDate;
  const header = document.createElement("h3");
  header.textContent = sighting.sightingDate;
  card.appendChild(header);
  const columns = document.createElement("div");
  columns.className = "side-columns";
  columns.appendChild(renderSideColumn(sighting, "left"));
  columns.appendChild(renderSideColumn(sighting, "right"));
  card.appendChild(columns);
  return card;
}

function renderElephantPanel(view) {
  elephantPanel.innerHTML = "";
  const heading = document.createElement("h2");
  heading.textContent =
    `${view.identity} — ${view.qualifyingCount} qualifying sightings`;
  elephantPanel.appendChild(heading);
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
  const payload = await postJSON("/api/pick", {
    identity: selectedIdentity,
    sightingDate,
    side,
    candidateId,
  });
  replaceSighting(payload);
}

refreshElephants().catch((error) => {
  scanStatus.textContent = `Error: ${error.message}`;
});

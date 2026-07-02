const els = {
  subtitle: document.getElementById("subtitle"),
  progress: document.getElementById("progress"),
  identityName: document.getElementById("identityName"),
  identityMeta: document.getElementById("identityMeta"),
  doneBanner: document.getElementById("doneBanner"),
  statusLine: document.getElementById("statusLine"),
  grid: document.getElementById("grid"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  doneBtn: document.getElementById("doneBtn"),
  footerLeft: document.getElementById("footerLeft"),
  footerRight: document.getElementById("footerRight"),
  toasts: document.getElementById("toasts"),
};

let appState = null;
let currentSide = "left";
let currentIdentity = null;
let busy = false;
let waitPollTimer = null;

async function getJson(url) {
  const res = await fetch(url);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed ${res.status}`);
  return data;
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed ${res.status}`);
  return data;
}

function showToast(message, variant = "") {
  const el = document.createElement("div");
  el.className = `toast ${variant}`.trim();
  el.textContent = message;
  els.toasts.appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 160);
  }, 2200);
}

function setBusy(value, message = "") {
  busy = value;
  document.body.classList.toggle("busy", value);
  if (message) els.statusLine.textContent = message;
}

function stateForSide(side) {
  return appState?.sides?.[side] || {};
}

function syncChrome() {
  document.querySelectorAll("[data-side]").forEach((btn) => {
    btn.classList.toggle("side-tab-active", btn.dataset.side === currentSide);
  });
  const left = stateForSide("left");
  const right = stateForSide("right");
  els.progress.textContent =
    `Left ${left.doneIdentities || 0}/${left.targetDoneIdentities || 100} · ` +
    `Right ${right.doneIdentities || 0}/${right.targetDoneIdentities || 100}`;
  const pool = appState?.pool || {};
  const scan = appState?.queueScan || {};
  const scanState = scan.running ? "scanning" : "scan complete";
  const current = scan.current ? ` · current ${scan.current}` : "";
  const errors = scan.futureError || (scan.errors || [])[0];
  const errorText = errors ? ` · latest error ${errors}` : "";
  els.subtitle.textContent =
    `${pool.eligibleIdentities || 0} eligible · ` +
    `${scan.ready || 0} ready collected · ` +
    `${scanState} ${scan.scanned || 0}/${scan.poolSize || 0}${current}${errorText} · ` +
    `manifest ${appState?.manifestPath || ""}`;
}

async function loadState() {
  appState = await getJson("/api/state");
  syncChrome();
  currentIdentity = stateForSide(currentSide).identity;
}

function candidateCropUrl(candidate) {
  const qs = new URLSearchParams({
    side: currentSide,
    identity: currentIdentity,
    candidateId: candidate.candidateId,
  });
  return `/api/crop?${qs.toString()}`;
}

function renderIdentity(payload) {
  clearTimeout(waitPollTimer);
  currentIdentity = payload.identity;
  els.identityName.textContent = `${payload.identity || ""} · ${payload.side}`;
  const minSelections = payload.minSelections;
  const maxSelections = payload.maxSelections;
  const selectionRange = selectionRangeText(minSelections, maxSelections);
  const selectionSummary = sideSelectionSummary(payload);
  els.identityMeta.textContent =
    payload.pairReady
      ? `${payload.candidateCount || 0} accepted crops`
      : pairWaitingText(payload);
  els.footerLeft.textContent = !payload.pairReady
    ? `Waiting for an elephant with enough images and sightings on both sides.`
    : payload.done
      ? `${selectionSummary} · Exported. Adjust picks and click Overwrite to replace.`
      : `${selectionSummary} · Select ${selectionRange} images.`;
  els.footerRight.textContent = payload.status;
  const selectionReady =
    payload.selectedCount >= minSelections && payload.selectedCount <= maxSelections;
  els.doneBtn.disabled = !payload.pairReady || !selectionReady;
  els.doneBtn.textContent = payload.done ? "Overwrite" : "Done";

  els.doneBanner.hidden = !payload.bothDone;
  if (payload.bothDone) {
    els.doneBanner.textContent =
      "✓ This elephant is done — left and right are exported. Editing a side and clicking Overwrite replaces the saved selection.";
  }

  if (!payload.pairReady) {
    els.statusLine.textContent = pairWaitingText(payload);
    waitPollTimer = setTimeout(refreshWhileWaiting, 3000);
  } else if (payload.error) {
    els.statusLine.textContent = payload.error;
  } else if (payload.status === "insufficient") {
    els.statusLine.textContent =
      `Fewer than ${minSelections} accepted crops for this side. Move to another identity.`;
  } else {
    els.statusLine.textContent = payload.done
      ? `This side is exported. Re-check ${selectionRange} images and click Overwrite to replace it.`
      : `Review crops and check ${selectionRange} images.`;
  }

  els.grid.innerHTML = "";
  if (!payload.pairReady) return;
  for (const candidate of payload.candidates || []) {
    const labelText =
      `${candidate.photoIdentifier} · ${candidate.date} · ` +
      `${candidate.boxHeightWidth.toFixed(2)} h/w`;
    const card = document.createElement("label");
    card.className = candidate.selected ? "tile tile-selected" : "tile";

    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = !!candidate.selected;
    input.addEventListener("change", async () => {
      try {
        setBusy(true, "Updating selection...");
        const next = await postJson("/api/select", {
          side: currentSide,
          identity: currentIdentity,
          candidateId: candidate.candidateId,
          selected: input.checked,
        });
        renderIdentity(next);
      } catch (error) {
        input.checked = !input.checked;
        showToast(error.message, "toast-bad");
      } finally {
        setBusy(false);
      }
    });

    const img = document.createElement("img");
    img.loading = "lazy";
    img.alt = labelText;
    img.src = candidateCropUrl(candidate);

    const label = document.createElement("span");
    label.className = "tile-label";
    label.textContent = labelText;

    card.append(input, img, label);
    els.grid.appendChild(card);
  }
}

function pairWaitingText(payload) {
  const pair = payload.pairStatus || {};
  const left = sideReadinessText(pair.left);
  const right = sideReadinessText(pair.right);
  const rule = readinessRuleText(payload.readinessRule);
  if (pair.loading) return `Loading candidates · left ${left} · right ${right} · ${rule}`;
  return `Not enough candidates · left ${left} · right ${right} · ${rule}`;
}

function sideReadinessText(counts) {
  if (!counts) return "loading";
  return `${counts.imageCount} images, ${counts.sightingCount} sightings`;
}

function readinessRuleText(rule) {
  const minSightings = rule?.minSightings ?? 4;
  const minImages = rule?.minImages ?? 15;
  const fallbackImages = rule?.fallbackImages ?? 25;
  return `need at least ${minSightings} sightings and ${minImages} images, or ${fallbackImages} images`;
}

function selectionRangeText(minSelections, maxSelections) {
  if (minSelections === maxSelections) return String(minSelections);
  return `${minSelections}-${maxSelections}`;
}

function sideSelectionSummary(payload) {
  const selections = payload.sideSelections || {};
  const left = selections.left ?? 0;
  const right = selections.right ?? 0;
  return `Left ${left} selected · Right ${right} selected`;
}

async function refreshWhileWaiting() {
  if (busy) {
    waitPollTimer = setTimeout(refreshWhileWaiting, 3000);
    return;
  }
  try {
    appState = await getJson("/api/state");
    syncChrome();
    currentIdentity = stateForSide(currentSide).identity;
    await loadIdentity();
  } catch (error) {
    showToast(error.message, "toast-bad");
  }
}

async function loadIdentity() {
  const identity = currentIdentity || stateForSide(currentSide).identity;
  if (!identity) return;
  try {
    setBusy(true, `Analyzing ${identity} ${currentSide} ear candidates...`);
    const qs = new URLSearchParams({ side: currentSide, identity });
    const payload = await getJson(`/api/identity?${qs.toString()}`);
    renderIdentity(payload);
  } catch (error) {
    showToast(error.message, "toast-bad");
  } finally {
    setBusy(false);
  }
}

async function navigate(delta) {
  if (busy) return;
  try {
    setBusy(true, "Moving...");
    appState = await postJson("/api/nav", { side: currentSide, delta });
    syncChrome();
    currentIdentity = stateForSide(currentSide).identity;
    await loadIdentity();
  } catch (error) {
    showToast(error.message, "toast-bad");
  } finally {
    setBusy(false);
  }
}

async function markDone() {
  if (busy || !currentIdentity) return;
  try {
    setBusy(true, "Exporting selected originals...");
    const payload = await postJson("/api/done", {
      side: currentSide,
      identity: currentIdentity,
    });
    appState = payload.state;
    syncChrome();
    renderIdentity(payload.identity);
    showToast("Exported to outputs/high_quality", "toast-ok");
  } catch (error) {
    showToast(error.message, "toast-bad");
  } finally {
    setBusy(false);
  }
}

async function switchSide(side) {
  if (busy || side === currentSide) return;
  currentSide = side;
  currentIdentity = stateForSide(currentSide).identity;
  syncChrome();
  await loadIdentity();
}

document.querySelectorAll("[data-side]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    await switchSide(btn.dataset.side || "left");
  });
});

els.prevBtn.addEventListener("click", () => navigate(-1));
els.nextBtn.addEventListener("click", () => navigate(1));
els.doneBtn.addEventListener("click", () => markDone());

function isTextEntryTarget(target) {
  if (!target) return false;
  if (target.isContentEditable) return true;
  if (target.tagName === "TEXTAREA") return true;
  if (target.tagName !== "INPUT") return false;
  return !["button", "checkbox", "radio", "range", "submit"].includes(target.type);
}

document.addEventListener("keydown", (event) => {
  if (
    event.key === "Tab" &&
    !event.altKey &&
    !event.ctrlKey &&
    !event.metaKey &&
    !isTextEntryTarget(event.target)
  ) {
    event.preventDefault();
    switchSide(currentSide === "left" ? "right" : "left");
    return;
  }
  if (isTextEntryTarget(event.target)) return;
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    navigate(-1);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    navigate(1);
  } else if (event.key === "1") {
    switchSide("left");
  } else if (event.key === "2") {
    switchSide("right");
  }
});

(async function boot() {
  try {
    await loadState();
    await loadIdentity();
  } catch (error) {
    showToast(error.message, "toast-bad");
  }
})();

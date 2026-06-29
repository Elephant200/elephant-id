const els = {
  subtitle: document.getElementById("subtitle"),
  progress: document.getElementById("progress"),
  identityName: document.getElementById("identityName"),
  identityMeta: document.getElementById("identityMeta"),
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
  const scanState = scan.running
    ? `scanning ${scan.processed || 0}/${scan.queueSize || 0}`
    : `scan complete ${scan.processed || scan.pairCached || 0}/${scan.queueSize || 0}`;
  const current = scan.current ? ` · current ${scan.current}` : "";
  const errors = scan.futureError || (scan.errors || [])[0];
  const errorText = errors ? ` · latest error ${errors}` : "";
  els.subtitle.textContent =
    `${pool.eligibleIdentities || 0} eligible identities · ` +
    `${scan.pairReady || 0}/${scan.queueSize || 0} ready pairs · ` +
    `${scan.pairCached || 0}/${scan.queueSize || 0} cached pairs · ` +
    `${scanState}${current}${errorText} · manifest ${appState?.manifestPath || ""}`;
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
  const minSelections = payload.minSelections || 4;
  const maxSelections = payload.maxSelections || 5;
  els.identityMeta.textContent =
    payload.pairReady
      ? `${payload.candidateCount || 0} accepted crops · ` +
        `${payload.selectedCount || 0}/${minSelections} selected`
      : pairWaitingText(payload);
  els.footerLeft.textContent = !payload.pairReady
    ? "Waiting for a 25-left / 25-right ready elephant."
    : payload.done
      ? "Exported"
      : `Select exactly ${minSelections} images.`;
  els.footerRight.textContent = payload.status;
  const selectionReady =
    payload.selectedCount >= minSelections && payload.selectedCount <= maxSelections;
  els.doneBtn.disabled = !payload.pairReady || payload.done || !selectionReady;

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
      ? "This identity is already in the high-quality manifest."
      : `Review crops and check exactly ${minSelections} images.`;
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
    input.disabled = !!payload.done;
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
  const left = pair.leftCount == null ? "loading" : String(pair.leftCount);
  const right = pair.rightCount == null ? "loading" : String(pair.rightCount);
  const min = payload.minSideCandidates || 25;
  if (pair.loading) return `Loading candidates · left ${left}/${min} · right ${right}/${min}`;
  return `Not enough candidates · left ${left}/${min} · right ${right}/${min}`;
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

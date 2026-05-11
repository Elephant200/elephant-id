async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) throw new Error(`Request failed ${res.status}`);
  return await res.json();
}

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed ${res.status}`);
  return await res.json();
}

const els = {
  subtitle: document.getElementById("subtitle"),
  grid: document.getElementById("grid"),
  footerLeft: document.getElementById("footerLeft"),
  footerRight: document.getElementById("footerRight"),
  done: document.getElementById("done"),
  doneSub: document.getElementById("doneSub"),
  restartBtn: document.getElementById("restartBtn"),
  toasts: document.getElementById("toasts"),
  elephantToggleBtn: document.getElementById("elephantToggleBtn"),
  filterModal: document.getElementById("filterModal"),
  helpQueue: document.getElementById("helpQueue"),
  helpSaved: document.getElementById("helpSaved"),
  browseEmpty: document.getElementById("browseEmpty"),
  yearExtentHint: document.getElementById("yearExtentHint"),
  shuffleToggleBtn: document.getElementById("shuffleToggleBtn"),
  imageModal: document.getElementById("imageModal"),
  modalImage: document.getElementById("modalImage"),
  modalTitle: document.getElementById("imageModalTitle"),
  sam3PresetSeg: document.getElementById("sam3PresetSeg"),
  sam3Status: document.getElementById("sam3Status"),
  sam3RunBtn: document.getElementById("sam3RunBtn"),
  imageModalClose: document.getElementById("imageModalClose"),
};

let busy = false;
let lastState = null;
let browseMode = "queue";
let savedListCache = { sightings: [] };
let savedIndex = 0;
let savedSightingsPage = 0;
let savedSightingImageEntries = [];
let modalFiltersSnapshot = null;

const BROWSE_ORDER = ["queue", "sightings"];
/** Who narrowed to one elephant: null · manual · caps */
let elephantLockReason = null;
let capsLockSeenInitialized = false;
let lastCapsLockState = false;

function showToast(title, detail, variant) {
  const host = els.toasts;
  if (!host) return;

  const el = document.createElement("div");
  el.className = `toast ${variant ? `toast-${variant}` : ""}`.trim();

  const t1 = document.createElement("div");
  t1.className = "t1";
  t1.textContent = title;

  el.appendChild(t1);
  if (detail) {
    const t2 = document.createElement("div");
    t2.className = "t2";
    t2.textContent = detail;
    el.appendChild(t2);
  }

  host.appendChild(el);
  const ttl = 1400;
  setTimeout(() => {
    el.style.opacity = "0";
    el.style.transition = "opacity 140ms ease-out";
    setTimeout(() => el.remove(), 160);
  }, ttl);
}

function maybeToastFromEvent(evt) {
  if (!evt || !evt.type) return;
  const map = {
    priority_image_toggled: ["Priority copy toggled", "warn"],
    undo_priority_toggle: ["Undid priority toggle", "ok"],
    saved_sighting_removed: ["Removed sighting folder", "bad"],
    undo_saved_sighting_removed: ["Restored sighting folder", "ok"],
  };
  const row = map[evt.type];
  if (!row) return;
  const detail = evt.detail ? `${evt.detail}` : evt.name ? `${evt.name} · ${evt.date || ""}` : "";
  showToast(row[0], detail.trim(), row[1]);
}

function setBusy(v) {
  busy = v;
  document.body.style.cursor = v ? "progress" : "";
}

function filenameFromPath(p) {
  const parts = (p || "").split("/");
  return parts[parts.length - 1] || p;
}

function setStats(line1, line2 = "") {
  const l1 = document.getElementById("statsLine1");
  const l2 = document.getElementById("statsLine2");
  if (l1) l1.textContent = line1 ?? "";
  if (l2) l2.textContent = line2 ?? "";
}

function formatSightingTitle(name, date, seekCode) {
  if (!name && !date) return "";
  const mid = `${name} · ${date}`;
  const s = (seekCode || "").trim();
  return s ? `${mid} · ${s}` : mid;
}

function formatSavedFolderTitle(folder) {
  const m = folder.match(/^(.+)__(\d{4}-\d{2}-\d{2})(?:__dup\d+)?$/);
  if (!m) return folder;
  return `${m[1]} · ${m[2]}`;
}

function normalizeImageRef(entry) {
  if (typeof entry === "string") return { path: entry, priorityStarred: false };
  return {
    path: entry?.path || "",
    priorityStarred: !!entry?.priorityStarred,
  };
}

function ensureThumbStar(card) {
  if (card.querySelector(".thumb-star")) return;
  const s = document.createElement("span");
  s.className = "thumb-star";
  s.textContent = "★";
  s.title = "Priority copy under samples";
  card.appendChild(s);
}

function optYearNum(el) {
  const v = el?.value?.trim();
  if (!v) return null;
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? n : null;
}

function filtersPayloadFromServer(f) {
  const z = f || {};
  return {
    sex: { B: !!z.sex?.B, C: !!z.sex?.C },
    tusks: {
      both: !!z.tusks?.both,
      leftOnly: !!z.tusks?.leftOnly,
      rightOnly: !!z.tusks?.rightOnly,
      noTusks: !!z.tusks?.noTusks,
    },
    extreme: {
      left: !!z.extreme?.left,
      right: !!z.extreme?.right,
    },
    special: {
      leftEar: !!z.special?.leftEar,
      rightEar: !!z.special?.rightEar,
      body: !!z.special?.body,
    },
    yearMin: z.yearMin ?? null,
    yearMax: z.yearMax ?? null,
    nonNormalOnly: !!z.nonNormalOnly,
  };
}

function applyFiltersToModal(p) {
  const byId = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.checked = !!v;
  };
  byId("f-sex-B", p.sex.B);
  byId("f-sex-C", p.sex.C);
  byId("f-tusk-both", p.tusks.both);
  byId("f-tusk-left", p.tusks.leftOnly);
  byId("f-tusk-right", p.tusks.rightOnly);
  byId("f-tusk-none", p.tusks.noTusks);
  byId("f-x-left", p.extreme.left);
  byId("f-x-right", p.extreme.right);
  byId("f-sp-left", p.special.leftEar);
  byId("f-sp-right", p.special.rightEar);
  byId("f-sp-body", p.special.body);
  byId("f-non-normal", p.nonNormalOnly);
  const ymin = document.getElementById("f-year-min");
  const ymax = document.getElementById("f-year-max");
  if (ymin) ymin.value = p.yearMin != null ? String(p.yearMin) : "";
  if (ymax) ymax.value = p.yearMax != null ? String(p.yearMax) : "";
}

function collectFiltersPayload() {
  return {
    sex: {
      B: !!document.getElementById("f-sex-B")?.checked,
      C: !!document.getElementById("f-sex-C")?.checked,
    },
    tusks: {
      both: !!document.getElementById("f-tusk-both")?.checked,
      leftOnly: !!document.getElementById("f-tusk-left")?.checked,
      rightOnly: !!document.getElementById("f-tusk-right")?.checked,
      noTusks: !!document.getElementById("f-tusk-none")?.checked,
    },
    extreme: {
      left: !!document.getElementById("f-x-left")?.checked,
      right: !!document.getElementById("f-x-right")?.checked,
    },
    special: {
      leftEar: !!document.getElementById("f-sp-left")?.checked,
      rightEar: !!document.getElementById("f-sp-right")?.checked,
      body: !!document.getElementById("f-sp-body")?.checked,
    },
    yearMin: optYearNum(document.getElementById("f-year-min")),
    yearMax: optYearNum(document.getElementById("f-year-max")),
    nonNormalOnly: !!document.getElementById("f-non-normal")?.checked,
  };
}

function syncYearExtentHint() {
  const hint = els.yearExtentHint;
  if (!hint) return;
  const ext = lastState?.yearExtent;
  if (ext && ext.min != null && ext.max != null) {
    hint.textContent = `Dataset includes years ${ext.min}–${ext.max}. Leave blank to ignore.`;
  } else {
    hint.textContent = "";
  }
}

function openFilterModal() {
  const m = els.filterModal;
  if (!m) return;
  modalFiltersSnapshot = filtersPayloadFromServer(lastState?.filters);
  applyFiltersToModal(modalFiltersSnapshot);
  syncYearExtentHint();
  m.hidden = false;
}

function closeFilterModal(restoreSnapshot) {
  const m = els.filterModal;
  if (!m) return;
  m.hidden = true;
  if (restoreSnapshot && modalFiltersSnapshot) {
    applyFiltersToModal(modalFiltersSnapshot);
  }
  modalFiltersSnapshot = null;
}

const SAM3_DEFAULT_PRESET = "features";
let sam3Presets = null; // { presets: string[], default: string }
let sam3PresetsPromise = null;
let imageModalCtx = null; // { kind, path, preset, originalUrl, overlayUrl }
let sam3Running = false;
let clickTimer = null;

function loadSam3Presets() {
  if (sam3Presets) return Promise.resolve(sam3Presets);
  if (sam3PresetsPromise) return sam3PresetsPromise;
  sam3PresetsPromise = getJson("/api/sam3/presets")
    .then((data) => {
      sam3Presets = {
        presets: Array.isArray(data?.presets) && data.presets.length ? data.presets : [SAM3_DEFAULT_PRESET],
        default: data?.default || SAM3_DEFAULT_PRESET,
      };
      return sam3Presets;
    })
    .catch(() => {
      sam3Presets = { presets: [SAM3_DEFAULT_PRESET, "body"], default: SAM3_DEFAULT_PRESET };
      return sam3Presets;
    });
  return sam3PresetsPromise;
}

function renderSam3PresetChips() {
  const host = els.sam3PresetSeg;
  if (!host || !sam3Presets || !imageModalCtx) return;
  host.innerHTML = "";
  for (const p of sam3Presets.presets) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "seg-btn";
    b.textContent = p;
    if (p === imageModalCtx.preset) b.classList.add("seg-on");
    b.addEventListener("click", () => {
      imageModalCtx.preset = p;
      host.querySelectorAll(".seg-btn").forEach((el) => el.classList.toggle("seg-on", el === b));
    });
    host.appendChild(b);
  }
}

function setSam3Status(text, isError) {
  const s = els.sam3Status;
  if (!s) return;
  s.textContent = text || "";
  s.classList.toggle("is-error", !!isError);
}

function buildImageUrl(ctx) {
  if (ctx.kind === "samples") return `/image?samplesRel=${encodeURIComponent(ctx.path)}`;
  return `/image?p=${encodeURIComponent(ctx.path)}`;
}

function revokeImageModalUrls() {
  if (imageModalCtx?.overlayUrl) {
    try {
      URL.revokeObjectURL(imageModalCtx.overlayUrl);
    } catch {}
    imageModalCtx.overlayUrl = null;
  }
}

async function openImageModal({ kind, path, label }) {
  if (!els.imageModal) return;
  await loadSam3Presets();
  imageModalCtx = {
    kind,
    path,
    preset: sam3Presets?.default || SAM3_DEFAULT_PRESET,
    originalUrl: buildImageUrl({ kind, path }),
    overlayUrl: null,
  };
  if (els.modalTitle) els.modalTitle.textContent = label || path;
  if (els.modalImage) {
    els.modalImage.alt = label || "";
    els.modalImage.src = imageModalCtx.originalUrl;
  }
  renderSam3PresetChips();
  setSam3Status("");
  els.imageModal.hidden = false;
}

function closeImageModal() {
  if (!els.imageModal) return;
  revokeImageModalUrls();
  els.imageModal.hidden = true;
  imageModalCtx = null;
  setSam3Status("");
}

async function runSam3() {
  if (sam3Running || !imageModalCtx) return;
  sam3Running = true;
  setSam3Status(`Running SAM3 (${imageModalCtx.preset})...`);
  if (els.sam3RunBtn) els.sam3RunBtn.disabled = true;
  try {
    const body =
      imageModalCtx.kind === "samples"
        ? { samplesRel: imageModalCtx.path, preset: imageModalCtx.preset }
        : { imagePath: imageModalCtx.path, preset: imageModalCtx.preset };
    const res = await fetch("/api/sam3", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      let msg = `Request failed (${res.status})`;
      try {
        const j = await res.json();
        if (j?.error) msg = j.error;
      } catch {}
      setSam3Status(msg, true);
      return;
    }
    const blob = await res.blob();
    revokeImageModalUrls();
    const url = URL.createObjectURL(blob);
    imageModalCtx.overlayUrl = url;
    if (els.modalImage) els.modalImage.src = url;
    setSam3Status(`Overlay: ${imageModalCtx.preset}`);
  } catch (e) {
    setSam3Status(String(e?.message || e), true);
  } finally {
    sam3Running = false;
    if (els.sam3RunBtn) els.sam3RunBtn.disabled = false;
  }
}

function attachThumbModalOpener(card, openArgs) {
  card._modalArgs = openArgs;
  card.addEventListener("click", (e) => {
    if (e.detail > 1) return; // ignore the click leading a dblclick
    if (clickTimer) clearTimeout(clickTimer);
    clickTimer = setTimeout(() => {
      clickTimer = null;
      openImageModal(openArgs);
    }, 220);
  });
}

function navigateImageModal(delta) {
  if (!imageModalCtx || !els.grid) return;
  const cards = Array.from(els.grid.querySelectorAll(".thumb")).filter(
    (c) => c._modalArgs && c._modalArgs.kind === imageModalCtx.kind
  );
  if (!cards.length) return;
  const idx = cards.findIndex((c) => c._modalArgs.path === imageModalCtx.path);
  if (idx === -1) return;
  const next = idx + delta;
  if (next < 0 || next >= cards.length) return;
  openImageModal(cards[next]._modalArgs);
}

function cancelPendingThumbClick() {
  if (clickTimer) {
    clearTimeout(clickTimer);
    clickTimer = null;
  }
}

function tileMinForCount(n) {
  if (n <= 3) return 440;
  if (n <= 6) return 380;
  if (n <= 10) return 320;
  if (n <= 16) return 270;
  return 240;
}

function tileHeightForCount(n) {
  if (n <= 3) return 420;
  if (n <= 6) return 360;
  if (n <= 10) return 310;
  if (n <= 16) return 270;
  return 250;
}

/** Two stacked rows at the ~7-image density: 2 × (rowHeight + gap); rowHeight matches tileHeightForCount band for n≤10 */
function soloThumbTargetPx() {
  const rowH = tileHeightForCount(7);
  const gap = 10;
  return 2 * (rowH + gap);
}

function gridWrapContentWidth() {
  const wrap = els.grid?.parentElement;
  if (!wrap) return Infinity;
  return wrap.clientWidth;
}

/** If false, don’t enlarge single-row tiles — they would wrap and oscillate with ResizeObserver. */
function singleRowBumpFits(n, minW, gapPx = 10) {
  const w = gridWrapContentWidth();
  const total = n * minW + (n - 1) * gapPx;
  return total <= w + 1;
}

function clearGridLayoutStyles() {
  const grid = els.grid;
  if (!grid) return;
  grid.classList.remove("grid--solo", "grid--one-row", "grid--saved-starred");
  grid.style.removeProperty("--tile-min");
  grid.style.removeProperty("--tile-h");
  grid.style.removeProperty("--solo-h");
}

/** Starred sighting grid: dense packing around 2×2 priority cells; uses same --tile-min/--tile-h bands as the queue. */
function layoutSavedStarredGrid(n) {
  const grid = els.grid;
  if (!grid) return;
  grid.classList.remove("grid--solo", "grid--one-row");
  grid.style.removeProperty("--solo-h");
  grid.classList.add("grid--saved-starred");
  const count = Math.max(
    1,
    typeof n === "number" && Number.isFinite(n) ? Math.floor(n) : grid.children.length
  );
  grid.style.setProperty("--tile-min", `${tileMinForCount(count)}px`);
  grid.style.setProperty("--tile-h", `${tileHeightForCount(count)}px`);
}

function scheduleLayoutForGridIfNeeded() {
  const g = els.grid;
  if (!g?.children.length) return;
  if (browseMode === "sightings") layoutSavedStarredGrid(g.children.length);
  else scheduleGridThumbLayout(g.children.length);
}

function applyTileSizingBaseline(n) {
  const grid = els.grid;
  if (!grid) return;
  grid.classList.remove("grid--solo", "grid--one-row");
  if (n === 1) {
    grid.style.removeProperty("--tile-min");
    grid.style.removeProperty("--tile-h");
    const soloPx = soloThumbTargetPx();
    grid.style.setProperty("--solo-h", `${soloPx}px`);
    grid.classList.add("grid--solo");
    return;
  }
  grid.style.removeProperty("--solo-h");
  grid.style.setProperty("--tile-min", `${tileMinForCount(n)}px`);
  grid.style.setProperty("--tile-h", `${tileHeightForCount(n)}px`);
}

function countThumbRows(grid) {
  const cells = [...grid.children];
  if (!cells.length) return 0;
  return new Set(cells.map((c) => c.offsetTop)).size;
}

function bumpIfSingleRowLayout(n) {
  const grid = els.grid;
  if (!grid || n <= 1) return;
  grid.classList.remove("grid--one-row");

  if (countThumbRows(grid) !== 1) return;

  let minW = 340;
  let h = 372;
  if (n <= 4) {
    minW = 440;
    h = 430;
  } else if (n <= 7) {
    minW = 400;
    h = 400;
  }

  if (!singleRowBumpFits(n, minW)) return;

  grid.style.setProperty("--tile-min", `${minW}px`);
  grid.style.setProperty("--tile-h", `${h}px`);
  grid.classList.add("grid--one-row");

  requestAnimationFrame(() => {
    if (countThumbRows(grid) > 1) {
      grid.classList.remove("grid--one-row");
      grid.style.setProperty("--tile-min", `${tileMinForCount(n)}px`);
      grid.style.setProperty("--tile-h", `${tileHeightForCount(n)}px`);
    }
  });
}

function scheduleGridThumbLayout(n) {
  const grid = els.grid;
  if (!grid) return;
  if (!n) {
    clearGridLayoutStyles();
    return;
  }
  applyTileSizingBaseline(n);
  if (n === 1) return;
  requestAnimationFrame(() => bumpIfSingleRowLayout(n));
}

function syncPageSizeChips(pageSize) {
  const ps = pageSize ?? 18;
  document.querySelectorAll("button[data-pagesize]").forEach((btn) => {
    const v = parseInt(btn.getAttribute("data-pagesize"), 10);
    btn.classList.toggle("chip-on", v === ps);
  });
}

function syncShuffleToggleBtn(enabled) {
  const b = els.shuffleToggleBtn;
  if (!b) return;
  const on = enabled !== false;
  b.textContent = on ? "Shuffle: on" : "Shuffle: off";
  b.setAttribute("aria-pressed", on ? "true" : "false");
}

function updateBrowseChrome() {
  const q = browseMode === "queue";
  if (els.helpQueue) els.helpQueue.hidden = !q;
  if (els.helpSaved) els.helpSaved.hidden = q;
  if (els.shuffleToggleBtn)
    els.shuffleToggleBtn.hidden = !q || !!(lastState && lastState.done);
  document.querySelectorAll("[data-browse]").forEach((btn) => {
    btn.classList.toggle("seg-on", btn.getAttribute("data-browse") === browseMode);
  });
  const eb = els.elephantToggleBtn;
  if (eb) {
    const show = q && lastState && !lastState.done;
    eb.classList.toggle("tb-btn-concealed", !show);
    eb.toggleAttribute("disabled", !show);
    eb.setAttribute("aria-hidden", show ? "false" : "true");
  }
}

function renderQueue(state) {
  lastState = state;
  maybeToastFromEvent(state.event);
  updateBrowseChrome();

  const elephantBtn = els.elephantToggleBtn;
  if (elephantBtn && !state.done && state.name) {
    elephantBtn.textContent = state.elephantOnly ? "Show all elephants" : `Only · ${state.name}`;
  }

  if (state.done) {
    els.done.hidden = false;
    els.doneSub.textContent =
      "No sightings match the current filters (or the queue is empty). Copies stay under dataset/samples/.";
    els.grid.innerHTML = "";
    clearGridLayoutStyles();
    els.subtitle.textContent = "";
    setStats("", "");
    els.footerLeft.textContent = "";
    els.footerRight.textContent = "";
    if (els.browseEmpty) els.browseEmpty.hidden = true;
    syncShuffleToggleBtn(state.shuffleEnabled);
    syncPageSizeChips(state.pageSize);
    return;
  }

  els.done.hidden = true;
  els.subtitle.textContent = formatSightingTitle(state.name, state.date, state.seekCode);
  setStats(`Sighting ${state.queueIndex + 1} of ${state.remaining}`, `${state.imageCount} images · showing ${state.images.length}`);
  els.footerLeft.textContent = `Sighting ${state.queueIndex + 1} of ${state.remaining} · Page ${state.page + 1} / ${state.pages}`;
  els.footerRight.textContent = `${state.imageCount} images · showing ${state.images.length}`;
  if (els.browseEmpty) els.browseEmpty.hidden = true;

  const size = 420;
  els.grid.innerHTML = "";
  els.grid.classList.remove("grid--saved-starred");
  for (const raw of state.images || []) {
    const { path: p, priorityStarred } = normalizeImageRef(raw);
    if (!p) continue;
    const card = document.createElement("div");
    card.className = "thumb";
    attachThumbModalOpener(card, { kind: "coded", path: p, label: filenameFromPath(p) });
    card.addEventListener("dblclick", (e) => {
      e.preventDefault();
      cancelPendingThumbClick();
      act(() => postJson("/api/toggle_priority_image", { imagePath: p }));
    });
    const img = document.createElement("img");
    img.loading = "lazy";
    img.alt = filenameFromPath(p);
    img.src = `/thumb?p=${encodeURIComponent(p)}&s=${size}`;
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = filenameFromPath(p);
    card.appendChild(img);
    card.appendChild(label);
    if (priorityStarred) ensureThumbStar(card);
    els.grid.appendChild(card);
  }
  scheduleGridThumbLayout(els.grid.children.length);

  syncShuffleToggleBtn(state.shuffleEnabled);
  syncPageSizeChips(state.pageSize);
}

async function loadSavedLists() {
  const data = await getJson("/api/saved/list");
  savedListCache = {
    sightings: data.sightings || [],
  };
  return savedListCache;
}

async function loadSavedSightingImages(folderRel) {
  const res = await fetch(`/api/saved/sighting_images?rel=${encodeURIComponent(folderRel)}`);
  const data = await res.json();
  if (!res.ok) return [];
  return data.rels || [];
}

async function renderSavedBrowse(opts = {}) {
  const reloadList = opts.reloadList !== false;
  const reloadSightingImages = opts.reloadSightingImages !== false;
  updateBrowseChrome();
  els.done.hidden = true;

  if (browseMode !== "sightings") {
    return;
  }

  if (reloadList) await loadSavedLists();
  const list = savedListCache.sightings;
  if (savedIndex >= list.length) savedIndex = Math.max(0, list.length - 1);
  if (list.length === 0) {
    els.grid.innerHTML = "";
    clearGridLayoutStyles();
    els.subtitle.textContent = "Starred";
    setStats("", "");
    els.footerLeft.textContent = "";
    els.footerRight.textContent = "";
    if (els.browseEmpty) els.browseEmpty.hidden = false;
    syncPageSizeChips(lastState?.pageSize);
    return;
  }
  if (els.browseEmpty) els.browseEmpty.hidden = true;
  const cur = list[savedIndex];
  if (reloadSightingImages) {
    savedSightingImageEntries = await loadSavedSightingImages(cur.rel);
  }
  const sightLabel = formatSavedFolderTitle(cur.folder);
  els.subtitle.textContent = sightLabel;

  const ps = lastState?.pageSize ?? 18;
  const total = savedSightingImageEntries.length;
  const pages = Math.max(1, Math.ceil(total / ps));
  savedSightingsPage = Math.max(0, Math.min(savedSightingsPage, pages - 1));
  const start = savedSightingsPage * ps;
  const pageSlice = savedSightingImageEntries.slice(start, start + ps);

  setStats(`${savedIndex + 1} / ${list.length}`, `${total} images · showing ${pageSlice.length}`);
  els.footerLeft.textContent = `${sightLabel} · Sighting ${savedIndex + 1} of ${list.length} · Page ${savedSightingsPage + 1} / ${pages}`;
  els.footerRight.textContent = `${total} images · showing ${pageSlice.length}`;

  const size = 420;
  els.grid.innerHTML = "";
  for (const row of pageSlice) {
    const rel = typeof row === "string" ? row : row.rel;
    const priority = typeof row === "string" ? false : !!row.priority;
    if (!rel) continue;
    const card = document.createElement("div");
    card.className = priority ? "thumb thumb--saved-priority" : "thumb thumb--saved-context";
    const img = document.createElement("img");
    img.loading = "lazy";
    img.alt = filenameFromPath(rel);
    img.src = `/api/saved/file_thumb?rel=${encodeURIComponent(rel)}&s=${size}`;
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = filenameFromPath(rel);
    card.appendChild(img);
    card.appendChild(label);
    if (priority) ensureThumbStar(card);
    attachThumbModalOpener(card, { kind: "samples", path: rel, label: filenameFromPath(rel) });
    card.addEventListener("dblclick", (e) => {
      e.preventDefault();
      cancelPendingThumbClick();
      act(async () => {
        const v = await postJson("/api/toggle_priority_image", { samplesRel: rel });
        maybeToastFromEvent(v.event);
        return v;
      });
    });
    els.grid.appendChild(card);
  }
  layoutSavedStarredGrid(pageSlice.length);
  syncPageSizeChips(lastState?.pageSize);
}

async function setBrowseMode(mode) {
  browseMode = mode;
  savedIndex = 0;
  savedSightingsPage = 0;
  if (mode === "queue") {
    renderQueue(lastState || {});
    return;
  }
  setBusy(true);
  try {
    await renderSavedBrowse();
  } finally {
    setBusy(false);
  }
}

async function savedSightingsPageNav(delta) {
  if (busy || browseMode !== "sightings") return;
  const ps = lastState?.pageSize ?? 18;
  const total = savedSightingImageEntries.length;
  const pages = Math.max(1, Math.ceil(total / ps));
  savedSightingsPage = Math.max(0, Math.min(savedSightingsPage + delta, pages - 1));
  await renderSavedBrowse({ reloadList: false, reloadSightingImages: false });
}

async function savedNavigate(delta) {
  if (busy) return;
  if (browseMode !== "sightings") return;
  await loadSavedLists();
  const n = savedListCache.sightings.length;
  if (!n) return;
  savedSightingsPage = 0;
  savedIndex = (savedIndex + delta + n) % n;
  await renderSavedBrowse({ reloadList: false });
}

async function savedDeleteCurrent() {
  if (busy) return;
  if (browseMode !== "sightings") return;
  const list = savedListCache.sightings;
  if (!list.length || savedIndex >= list.length) return;
  const rel = list[savedIndex].rel;
  setBusy(true);
  try {
    const res = await fetch("/api/saved/remove", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: "sighting", rel }),
    });
    let data = {};
    try {
      data = await res.json();
    } catch {
      showToast("Remove failed", res.statusText || String(res.status), "bad");
      return;
    }
    if (!res.ok) {
      showToast("Remove failed", data.error || String(res.status), "bad");
      return;
    }
    maybeToastFromEvent(data.review?.event);
    lastState = data.review;
    await loadSavedLists();
    if (savedIndex >= savedListCache.sightings.length) {
      savedIndex = Math.max(0, savedListCache.sightings.length - 1);
    }
    savedSightingsPage = 0;
    await renderSavedBrowse();
  } catch (e) {
    console.error(e);
    showToast("Remove failed", String(e?.message || e), "bad");
  } finally {
    setBusy(false);
  }
}

async function refresh() {
  setBusy(true);
  try {
    const state = await getJson("/api/state");
    lastState = state;
    if (browseMode === "queue") {
      renderQueue(state);
    } else {
      await renderSavedBrowse();
    }
  } finally {
    setBusy(false);
  }
}

async function act(fn) {
  if (busy) return;
  setBusy(true);
  try {
    const v = await fn();
    lastState = v;
    if (browseMode === "queue") renderQueue(v);
    else await renderSavedBrowse();
  } catch (e) {
    console.error(e);
  } finally {
    setBusy(false);
  }
}

function isEditableTarget(t) {
  if (!t) return false;
  const tag = (t.tagName || "").toLowerCase();
  return tag === "input" || tag === "textarea" || t.isContentEditable;
}

function isFilterModalOpen() {
  return els.filterModal && !els.filterModal.hidden;
}

function isImageModalOpen() {
  return els.imageModal && !els.imageModal.hidden;
}

function isModalOpen() {
  return isFilterModalOpen() || isImageModalOpen();
}

function cycleBrowseMode(delta) {
  const i = BROWSE_ORDER.indexOf(browseMode);
  const next = BROWSE_ORDER[(i + delta + BROWSE_ORDER.length) % BROWSE_ORDER.length];
  void setBrowseMode(next);
}

function syncCapsLockFromKeyboard(e) {
  if (!e.getModifierState) return;
  const caps = e.getModifierState("CapsLock");
  if (!capsLockSeenInitialized) {
    capsLockSeenInitialized = true;
    lastCapsLockState = caps;
    void applyCapsLockElephantState(caps);
    return;
  }
  if (caps === lastCapsLockState) return;
  lastCapsLockState = caps;
  void applyCapsLockElephantState(caps);
}

async function applyCapsLockElephantState(caps) {
  if (busy) return;
  if (browseMode !== "queue" || !lastState || lastState.done) return;

  if (caps) {
    if (elephantLockReason === "manual") return;
    if (!lastState.elephantOnly) {
      elephantLockReason = "caps";
      setBusy(true);
      try {
        const v = await postJson("/api/elephant_only", { enabled: true });
        lastState = v;
        renderQueue(v);
      } catch (err) {
        console.error(err);
        elephantLockReason = null;
      } finally {
        setBusy(false);
      }
    }
    return;
  }

  if (elephantLockReason === "caps") {
    elephantLockReason = null;
    setBusy(true);
    try {
      const v = await postJson("/api/elephant_only", { enabled: false });
      lastState = v;
      renderQueue(v);
    } catch (err) {
      console.error(err);
    } finally {
      setBusy(false);
    }
  }
}

document.addEventListener("keydown", (e) => {
  if (isImageModalOpen()) {
    if (e.key === "Escape") {
      e.preventDefault();
      closeImageModal();
      return;
    }
    if (e.key === "Enter" && !isEditableTarget(e.target)) {
      e.preventDefault();
      void runSam3();
      return;
    }
    if (e.key === "ArrowLeft" && !isEditableTarget(e.target)) {
      e.preventDefault();
      navigateImageModal(-1);
      return;
    }
    if (e.key === "ArrowRight" && !isEditableTarget(e.target)) {
      e.preventDefault();
      navigateImageModal(1);
      return;
    }
    return;
  }

  if (busy) return;

  if (isFilterModalOpen()) {
    if (e.key === "Escape") {
      e.preventDefault();
      closeFilterModal(true);
    }
    return;
  }

  if (isEditableTarget(e.target)) return;

  syncCapsLockFromKeyboard(e);

  if (e.key === "Tab") {
    e.preventDefault();
    cycleBrowseMode(e.shiftKey ? -1 : 1);
    return;
  }

  if ((e.metaKey || e.ctrlKey) && (e.key === "z" || e.key === "Z")) {
    e.preventDefault();
    act(() => postJson("/api/undo", {}));
    return;
  }

  if (browseMode === "sightings") {
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      savedNavigate(-1);
      return;
    }
    if (e.key === "ArrowRight") {
      e.preventDefault();
      savedNavigate(1);
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      void savedSightingsPageNav(-1);
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      void savedSightingsPageNav(1);
      return;
    }
    if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault();
      void savedDeleteCurrent();
      return;
    }
    return;
  }

  const key = e.key;

  if (key === "ArrowLeft") {
    e.preventDefault();
    act(() => postJson("/api/nav", { delta: -1 }));
    return;
  }
  if (key === "ArrowRight") {
    e.preventDefault();
    act(() => postJson("/api/nav", { delta: 1 }));
    return;
  }
  if (key === "ArrowUp") {
    e.preventDefault();
    act(() => postJson("/api/page", { delta: -1 }));
    return;
  }
  if (key === "ArrowDown") {
    e.preventDefault();
    act(() => postJson("/api/page", { delta: 1 }));
    return;
  }

});

document.addEventListener("keyup", (e) => {
  if (busy) return;
  if (isModalOpen()) return;
  if (isEditableTarget(e.target)) return;
  syncCapsLockFromKeyboard(e);
});

document.querySelectorAll("button[data-pagesize]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const ps = parseInt(btn.getAttribute("data-pagesize"), 10);
    savedSightingsPage = 0;
    act(() => postJson("/api/page_size", { pageSize: ps }));
  });
});

els.shuffleToggleBtn?.addEventListener("click", () => {
  if (busy || browseMode !== "queue") return;
  const cur = !!lastState?.shuffleEnabled;
  act(() => postJson("/api/shuffle", { enabled: !cur }));
});

els.restartBtn?.addEventListener("click", () => {
  window.location.reload();
});

document.getElementById("openFilterModal")?.addEventListener("click", () => openFilterModal());

document.getElementById("filterApplyBtn")?.addEventListener("click", async () => {
  if (busy) return;
  const payload = collectFiltersPayload();
  closeFilterModal(false);
  setBusy(true);
  try {
    const v = await postJson("/api/filter", payload);
    savedSightingsPage = 0;
    if (browseMode === "sightings") savedIndex = 0;
    lastState = v;
    if (browseMode === "queue") renderQueue(v);
    else await renderSavedBrowse();
  } catch (e) {
    console.error(e);
  } finally {
    setBusy(false);
  }
});

document.getElementById("filterCancelBtn")?.addEventListener("click", () => {
  closeFilterModal(true);
});

els.filterModal?.querySelector(".modal-backdrop")?.addEventListener("click", () => {
  closeFilterModal(true);
});

els.imageModal?.querySelector(".modal-backdrop")?.addEventListener("click", () => {
  closeImageModal();
});

els.imageModalClose?.addEventListener("click", () => closeImageModal());

els.sam3RunBtn?.addEventListener("click", () => {
  void runSam3();
});

els.elephantToggleBtn?.addEventListener("click", () => {
  const en = !(lastState && lastState.elephantOnly);
  elephantLockReason = en ? "manual" : null;
  act(() => postJson("/api/elephant_only", { enabled: en }));
});

document.querySelectorAll("[data-browse]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const mode = btn.getAttribute("data-browse") || "queue";
    setBrowseMode(mode);
  });
});

window.addEventListener("resize", () => {
  scheduleLayoutForGridIfNeeded();
});

refresh();

const gridWrap = document.querySelector(".grid-wrap");
if (gridWrap && typeof ResizeObserver !== "undefined") {
  let roT = 0;
  const ro = new ResizeObserver(() => {
    clearTimeout(roT);
    roT = setTimeout(() => {
      scheduleLayoutForGridIfNeeded();
    }, 120);
  });
  ro.observe(gridWrap);
}

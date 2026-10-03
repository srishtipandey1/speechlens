"use strict";

const $ = (selector) => document.querySelector(selector);
const dimensionInfo = [
  { key: "pacing", label: "Pacing", note: "How closely word durations match the ideal or DEV reference." },
  { key: "pausing_fluency", label: "Pausing / fluency", note: "Excess inter-word pauses relative to the comparison." },
  { key: "intonation", label: "Intonation", note: "How closely voiced F0 spread matches the comparison." },
  { key: "energy", label: "Energy", note: "Additional downward speech-level change relative to the comparison." },
  { key: "articulation", label: "Articulation", note: "Syllable-nuclei rate and available transcript-count agreement." },
  { key: "disfluency", label: "Disfluency", note: "Excess voiced, speech-active time not assigned to aligned words." },
];
const flawTypes = [
  "pace_fast", "pace_slow", "long_pause", "monotone", "volume_dropoff", "filler", "stumble_repeat",
];
const flawLabels = {
  pace_fast: "Fast pace",
  pace_slow: "Slow pace",
  long_pause: "Long pause",
  monotone: "Monotone",
  volume_dropoff: "Volume dropoff",
  filler: "Filler",
  stumble_repeat: "Stumble / repeat",
};
const flawColors = {
  pace_fast: "#cf5948",
  pace_slow: "#b57b1c",
  long_pause: "#4478a8",
  monotone: "#8d5d42",
  volume_dropoff: "#567d38",
  filler: "#9a5b8d",
  stumble_repeat: "#627889",
};
const state = {
  demoIds: [],
  demoByLadder: {},
  currentId: "",
  currentResult: null,
  currentAudio: null,
  currentObjectUrl: null,
  wavesurfer: null,
  waveformReady: false,
  duration: 0,
  activeRegion: -1,
  metrics: null,
  animatedScore: 0,
  cdnNotice: "",
  showLowReliability: false,
  transcript: "",
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function formatNumber(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "unavailable";
}

function formatTime(value) {
  const total = Math.max(0, Number(value) || 0);
  const minutes = Math.floor(total / 60);
  return `${String(minutes).padStart(2, "0")}:${(total % 60).toFixed(1).padStart(4, "0")}`;
}

function setBanner(message, kind = "notice") {
  if (!message && kind === "notice" && state.cdnNotice) message = state.cdnNotice;
  const notice = $("#notice");
  const error = $("#error-banner");
  notice.hidden = kind !== "notice" || !message;
  error.hidden = kind !== "error" || !message;
  if (kind === "notice") notice.textContent = message || "";
  else error.textContent = message || "";
}

function responseMessage(payload, fallback) {
  const detail = payload?.detail;
  if (typeof detail === "string") return detail;
  if (detail && typeof detail.message === "string") return detail.message;
  if (typeof payload?.message === "string") return payload.message;
  return fallback;
}

async function getJson(url) {
  const response = await fetch(url);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(responseMessage(payload, `Request failed (${response.status})`));
  return payload;
}

function setTab(tab) {
  const demoActive = tab === "demo";
  $("#tab-demo").classList.toggle("is-active", demoActive);
  $("#tab-own").classList.toggle("is-active", !demoActive);
  $("#tab-demo").setAttribute("aria-selected", String(demoActive));
  $("#tab-own").setAttribute("aria-selected", String(!demoActive));
  $("#panel-demo").hidden = !demoActive;
  $("#panel-own").hidden = demoActive;
}

function prettyId(recordingId) {
  const match = recordingId.match(/__(ideal|L[1-5]|control_.+)$/i);
  const base = recordingId.split("__")[0];
  if (!match) return recordingId;
  const suffix = match[1].toLowerCase();
  if (suffix === "ideal") return `${base} · IDEAL`;
  if (/^l[1-5]$/.test(suffix)) return `${base} · ${suffix.toUpperCase()}`;
  return `${base} · ${suffix.replace("control_", "control: ").replaceAll("_", " ")}`;
}

function ladderPrefix(recordingId) {
  return recordingId.replace(/__(?:ideal|L[1-5]|control_.+)$/i, "");
}

function setLadderFor(recordingId) {
  const prefix = ladderPrefix(recordingId);
  state.demoByLadder = {};
  for (const id of state.demoIds) {
    if (ladderPrefix(id) !== prefix) continue;
    const match = id.match(/__(ideal|L[1-5])$/i);
    if (match) state.demoByLadder[match[1].toLowerCase()] = id;
  }
  const slider = $("#severity-slider");
  const selectorSuffix = recordingId.slice(prefix.length + 2).toLowerCase();
  const ladderIndex = selectorSuffix === "ideal" ? 0 : /^l[1-5]$/.test(selectorSuffix) ? Number(selectorSuffix.slice(1)) : -1;
  slider.disabled = Object.keys(state.demoByLadder).length < 2;
  if (ladderIndex >= 0) slider.value = String(ladderIndex);
  $("#severity-value").textContent = ladderIndex >= 0 ? (ladderIndex === 0 ? "IDEAL" : `L${ladderIndex}`) : "—";
}

function setLoading(isLoading) {
  $("#results").hidden = !isLoading && !state.currentResult;
  $("#loading-skeleton").hidden = !isLoading;
  $("#result-content").hidden = isLoading || !state.currentResult;
  $("#initial-state").hidden = isLoading || Boolean(state.currentResult);
}

function scoreModeLabel(mode) {
  return mode === "paired" ? "Paired" : "Reference-free";
}

function scoreBand(score) {
  if (score >= 80) return "Close to reference";
  if (score >= 60) return "Noticeable differences";
  return "Substantial differences";
}

function isReliableRegion(region) {
  return ["high", "medium"].includes(region.reliability?.level);
}

function visibleRegionEntries(result) {
  return (result.regions || []).map((region, index) => ({ region, index })).filter(({ region }) => (
    isReliableRegion(region) || state.showLowReliability
  ));
}

function updateLowToggle(result) {
  const lowCount = (result.regions || []).filter((region) => !isReliableRegion(region)).length;
  const checkbox = $("#show-low-regions");
  checkbox.checked = state.showLowReliability;
  checkbox.disabled = lowCount === 0;
  $("#low-toggle-label").textContent = `Show low-reliability regions (${lowCount})`;
}

function findingDeviation(region) {
  const observed = Number(region.observed_numeric);
  const expected = Number(region.expected_numeric);
  if (Number.isFinite(observed) && Number.isFinite(expected)) return Math.abs(observed - expected);
  const ratio = Number(region.features?.peak_rate_ratio);
  if (Number.isFinite(ratio)) return Math.abs(ratio - 1);
  const value = Number(region.z_score_or_ratio);
  return Number.isFinite(value) ? Math.abs(value) : Number(region.severity) || 0;
}

function renderSummary(result) {
  const total = Number(result.total_score ?? 0);
  const band = $("#score-band");
  band.textContent = scoreBand(total);
  band.className = `score-band ${total >= 80 ? "close" : total >= 60 ? "noticeable" : "substantial"}`;
  const findings = (result.regions || []).map((region, index) => ({ region, index })).sort((left, right) => {
    const reliabilityRank = (region) => ({ high: 0, medium: 1, low: 2 }[region.reliability?.level] ?? 3);
    return reliabilityRank(left.region) - reliabilityRank(right.region)
      || findingDeviation(right.region) - findingDeviation(left.region)
      || Number(left.region.start_s) - Number(right.region.start_s);
  }).slice(0, 3);
  const list = $("#summary-findings-list");
  if (!findings.length) {
    list.innerHTML = '<li class="empty-state">No flaw regions were detected.</li>';
    return;
  }
  list.innerHTML = findings.map(({ region, index }) => `
    <li><button class="finding-link" type="button" data-region-index="${index}">
      <span class="finding-time">${formatTime(region.start_s)}</span>
      <span class="finding-copy"><strong>${escapeHtml(region.sentence || "Timing differs from the reference here.")}</strong><small>${escapeHtml(region.suggestion || "Review this delivery segment.")}</small></span>
      <span class="finding-reliability ${escapeHtml(region.reliability?.level || "unavailable")}">${escapeHtml(region.reliability?.level || "unavailable")}</span>
    </button></li>`).join("");
  list.querySelectorAll("button[data-region-index]").forEach((button) => {
    button.addEventListener("click", () => {
      const index = Number(button.dataset.regionIndex);
      const region = result.regions[index];
      if (!isReliableRegion(region) && !state.showLowReliability) {
        state.showLowReliability = true;
        refreshFilteredRegions();
      }
      seekTo(Number(region.start_s), index);
    });
  });
}

function refreshFilteredRegions() {
  const result = state.currentResult;
  if (!result) return;
  updateLowToggle(result);
  renderRegions(result);
  renderWaveRegions(result);
  renderTranscript(result, state.transcript);
  renderTimelines(result);
}

function animateTotal(target) {
  const startValue = state.animatedScore;
  const startTime = performance.now();
  const duration = 480;
  function frame(now) {
    const fraction = Math.min(1, (now - startTime) / duration);
    const eased = 1 - (1 - fraction) ** 3;
    const value = startValue + (target - startValue) * eased;
    $("#summary-total-score").textContent = value.toFixed(1);
    if (fraction < 1) requestAnimationFrame(frame);
    else state.animatedScore = target;
  }
  requestAnimationFrame(frame);
}

function renderDimensions(result) {
  const values = result.scores || {};
  const root = $("#dimension-list");
  root.replaceChildren();
  for (const dimension of dimensionInfo) {
    const score = Number(values[dimension.key] ?? 0);
    const row = document.createElement("div");
    row.className = "dimension-row";
    row.innerHTML = `
      <div class="dimension-label"><strong>${escapeHtml(dimension.label)}</strong><small>${escapeHtml(dimension.note)}</small></div>
      <div class="dimension-track" role="meter" aria-label="${escapeHtml(dimension.label)} score" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${score}"><span class="dimension-fill"></span></div>
      <span class="dimension-value">${formatNumber(score, 0)}</span>`;
    root.append(row);
    requestAnimationFrame(() => { row.querySelector(".dimension-fill").style.width = `${Math.max(0, Math.min(100, score))}%`; });
  }
  animateTotal(Number(result.total_score ?? 0));
  const note = result.mode === "reference_free"
    ? "Reference-free scores are estimates against DEV-IDEAL statistics. Pausing/fluency and disfluency have enabled detector support; pacing, intonation, energy, and articulation are limited."
    : "Paired mode compares the read with its same-text ideal. This direct comparison is preferred when a suitable reference recording is available.";
  $("#reference-note").textContent = note;
}

function renderRadar(result) {
  if (!window.Plotly) {
    $("#radar-chart").hidden = true;
    $("#radar-fallback").hidden = false;
    return;
  }
  $("#radar-chart").hidden = false;
  $("#radar-fallback").hidden = true;
  const categories = dimensionInfo.map((dimension) => dimension.label);
  const values = dimensionInfo.map((dimension) => Number(result.scores?.[dimension.key] ?? 0));
  const radarDraw = window.Plotly.react("radar-chart", [{
    type: "scatterpolar", r: [...values, values[0]], theta: [...categories, categories[0]],
    fill: "toself", fillcolor: "rgba(8,127,120,.16)", line: { color: "#087f78", width: 2 },
    marker: { color: "#c65645", size: 5 }, hovertemplate: "%{theta}: %{r:.1f}<extra></extra>",
  }], {
    margin: { t: 16, r: 35, b: 28, l: 35 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent",
    showlegend: false, polar: { bgcolor: "transparent", radialaxis: { range: [0, 100], tickvals: [0, 50, 100], tickfont: { size: 9, color: "#607271" }, gridcolor: "#d8e3df", linecolor: "#d8e3df" }, angularaxis: { tickfont: { size: 10, color: "#172d2d" }, gridcolor: "#d8e3df" } },
  }, { displayModeBar: false, responsive: true });
  radarDraw.then(() => window.Plotly.Plots.resize("radar-chart"));
}

function timelineDuration(result) {
  const participant = result.series?.participant || {};
  const times = [
    ...(participant.f0_semitones || []).map((item) => Number(item.time_s) || 0),
    ...(participant.energy_db || []).map((item) => Number(item.time_s) || 0),
    ...(participant.speech_rate_sps || []).map((item) => Number(item.end_s) || 0),
    ...(result.regions || []).map((item) => Number(item.end_s) || 0),
  ];
  return Math.max(0.1, ...times);
}

function seekTo(seconds, selectedIndex = -1) {
  state.activeRegion = selectedIndex;
  highlightRegion(selectedIndex);
  if (state.currentAudio) {
    state.currentAudio.currentTime = Math.max(0, seconds);
    state.currentAudio.play().catch(() => {});
  }
  if (state.waveformReady && state.wavesurfer && state.duration > 0) {
    state.wavesurfer.setTime(Math.max(0, Math.min(state.duration, seconds)));
  }
  syncTranscript(seconds);
}

function regionColor(type) {
  return flawColors[type] || "#65788a";
}

function renderWaveRegions(result) {
  const container = $("#wave-regions");
  const regions = visibleRegionEntries(result);
  container.replaceChildren();
  for (const { region, index } of regions) {
    const start = Number(region.start_s) || 0;
    const end = Number(region.end_s) || start;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "wave-region";
    button.style.left = `${Math.max(0, start / state.duration * 100)}%`;
    button.style.width = `${Math.max(1.2, (end - start) / state.duration * 100)}%`;
    button.style.setProperty("--type-color", regionColor(region.type));
    button.style.backgroundColor = `${regionColor(region.type)}22`;
    button.style.color = regionColor(region.type);
    button.style.opacity = isReliableRegion(region) ? "1" : "0.32";
    button.textContent = flawLabels[region.type] || region.type;
    button.title = `${formatTime(start)}–${formatTime(end)} · ${flawLabels[region.type] || region.type}`;
    button.setAttribute("aria-label", `Seek to ${flawLabels[region.type] || region.type} region at ${formatTime(start)}`);
    button.addEventListener("click", () => seekTo(start, index));
    container.append(button);
  }
  const legend = $("#flaw-legend");
  legend.innerHTML = flawTypes.map((type) => `<span class="legend-item"><i class="legend-swatch" style="background:${regionColor(type)}"></i>${escapeHtml(flawLabels[type])}</span>`).join("");
}

function renderTranscript(result, transcriptText = "") {
  const container = $("#transcript-words");
  const alignedWords = result.series?.participant?.speech_rate_sps || [];
  const words = alignedWords.length ? alignedWords : transcriptText.trim().split(/\s+/).filter(Boolean).map((word) => ({ word }));
  if (!words.length) {
    container.innerHTML = '<p class="empty-state">No aligned word timings were returned.</p>';
    return;
  }
  container.replaceChildren();
  words.forEach((item, index) => {
    const word = document.createElement("button");
    word.type = "button";
    word.className = "word-chip";
    word.textContent = item.word;
    word.dataset.start = String(item.start_s ?? "");
    word.dataset.end = String(item.end_s ?? "");
    const start = Number(item.start_s);
    const end = Number(item.end_s);
    const inRegion = visibleRegionEntries(result).some(({ region }) => (
      start < Number(region.end_s) && end > Number(region.start_s)
    ));
    if (inRegion) word.classList.add("is-flawed");
    word.setAttribute("aria-label", Number.isFinite(start) ? `Seek to ${item.word} at ${formatTime(start)}` : item.word);
    word.addEventListener("click", () => { if (Number.isFinite(start)) seekTo(start); });
    container.append(word);
    if (index < words.length - 1) container.append(document.createTextNode(" "));
  });
}

function syncTranscript(seconds) {
  let current = null;
  document.querySelectorAll(".word-chip").forEach((button) => {
    const start = Number(button.dataset.start);
    const end = Number(button.dataset.end);
    const active = Number.isFinite(start) && seconds >= start && seconds <= end;
    button.classList.toggle("is-current", active);
    if (active) current = button;
  });
  if (current) current.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
  $("#playback-time").textContent = formatTime(seconds);
}

function highlightRegion(index) {
  document.querySelectorAll("#region-rows tr[data-region-index]").forEach((row) => {
    row.classList.toggle("is-selected", Number(row.dataset.regionIndex) === index);
  });
}

function reliabilityTitle(reliability) {
  const precision = reliability?.test_precision;
  const recall = reliability?.test_recall;
  if (precision == null || recall == null) return "Held-out test precision and recall unavailable";
  return `Held-out TEST precision: ${Number(precision).toFixed(6)}; recall: ${Number(recall).toFixed(6)}; IoU threshold: ${Number(reliability.iou_threshold ?? 0.3).toFixed(1)}`;
}

function renderRegions(result) {
  const rows = $("#region-rows");
  const regions = visibleRegionEntries(result);
  $("#region-count").textContent = String(regions.length);
  if (!regions.length) {
    rows.innerHTML = '<tr><td colspan="5" class="empty-state">No high- or medium-reliability regions are shown.</td></tr>';
    return;
  }
  rows.innerHTML = regions.map(({ region, index }) => {
    const type = String(region.type || "other_deviation");
    const reliability = region.reliability || {};
    const level = ["high", "medium", "low"].includes(reliability.level) ? reliability.level : "unavailable";
    const phrase = (region.words || []).join(" ");
    const message = [region.sentence, region.suggestion].filter(Boolean).join(" ");
    const measured = [
      region.observed_value || "Observed value unavailable",
      `Expected: ${region.expected_value || "unavailable"}`,
      `Ratio / z: ${formatNumber(region.z_score_or_ratio, 2)}`,
      `Formula: ${region.formula || "unavailable"}`,
    ].join(" · ");
    return `<tr data-region-index="${index}">
      <td><button class="time-link" type="button" data-seek="${Number(region.start_s) || 0}" data-region="${index}">${formatTime(region.start_s)}–${formatTime(region.end_s)}</button></td>
      <td><span class="type-tag" style="--type-color:${regionColor(type)}">${escapeHtml(flawLabels[type] || type)}</span></td>
      <td class="measured-cell">${escapeHtml(measured)}</td>
      <td class="explanation-cell">${phrase ? `<strong>${escapeHtml(phrase)}</strong>` : ""}<span>${escapeHtml(message || "Explanation unavailable")}</span></td>
      <td><span class="reliability-badge ${level}" title="${escapeHtml(reliabilityTitle(reliability))}">${level}</span></td>
    </tr>`;
  }).join("");
  rows.querySelectorAll("button[data-seek]").forEach((button) => {
    button.addEventListener("click", () => seekTo(Number(button.dataset.seek), Number(button.dataset.region)));
  });
}

function seriesToTrace(points, valueKey = "value", xKey = "time_s") {
  return {
    x: points.map((point) => Number(point[xKey]) || 0),
    y: points.map((point) => point[valueKey] == null ? null : Number(point[valueKey])),
  };
}

function renderTimelines(result) {
  if (!window.Plotly) {
    $("#timeseries-chart").hidden = true;
    $("#timeseries-fallback").hidden = false;
    return;
  }
  $("#timeseries-chart").hidden = false;
  $("#timeseries-fallback").hidden = true;
  const participant = result.series?.participant || {};
  const baseline = result.series?.baseline || {};
  const traces = [];
  const measures = [
    ["F0 · semitones", participant.f0_semitones, baseline.f0_semitones, "y", "x", "#087f78"],
    ["Energy · dB", participant.energy_db, baseline.energy_db, "y2", "x2", "#c65645"],
    ["Speech rate · syllables/s", participant.speech_rate_sps, baseline.speech_rate_sps, "y3", "x3", "#4478a8"],
  ];
  for (const [label, actual, reference, yaxis, xaxis, color] of measures) {
    const participantTrace = seriesToTrace(actual || [], "value", actual?.[0]?.time_s === undefined ? "start_s" : "time_s");
    if (label.startsWith("Speech rate")) participantTrace.x = (actual || []).map((point) => (Number(point.start_s) + Number(point.end_s)) / 2);
    traces.push({ ...participantTrace, type: "scatter", mode: "lines+markers", name: `Participant · ${label}`, xaxis, yaxis, line: { color, width: 1.8 }, marker: { size: 3 }, connectgaps: false });
    if (reference?.length) {
      const baselineTrace = seriesToTrace(reference, "value", reference[0]?.time_s === undefined ? "start_s" : "time_s");
      if (label.startsWith("Speech rate")) baselineTrace.x = reference.map((point) => (Number(point.start_s) + Number(point.end_s)) / 2);
      traces.push({ ...baselineTrace, type: "scatter", mode: "lines", name: `Baseline · ${label}`, xaxis, yaxis, line: { color, width: 1.4, dash: "dot" }, connectgaps: false });
    }
  }
  const shapes = visibleRegionEntries(result).map(({ region }) => ({
    type: "rect", xref: "x", yref: "paper", x0: Number(region.start_s), x1: Number(region.end_s), y0: 0, y1: 1,
    fillcolor: regionColor(region.type), opacity: isReliableRegion(region) ? 0.10 : 0.035, line: { width: 0 }, layer: "below",
  }));
  shapes.push({ type: "line", xref: "x", yref: "paper", x0: 0, x1: 0, y0: 0, y1: 1, line: { color: "#172d2d", width: 1.5, dash: "dot" } });
  const timelineDraw = window.Plotly.react("timeseries-chart", traces, {
    height: 400, margin: { l: 62, r: 20, t: 38, b: 35 }, paper_bgcolor: "transparent", plot_bgcolor: "#ffffff",
    font: { family: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 10, color: "#607271" },
    grid: { rows: 3, columns: 1, pattern: "independent" },
    xaxis: { domain: [0, 1], anchor: "y", range: [0, state.duration], showticklabels: false, gridcolor: "#edf1ef", zeroline: false },
    xaxis2: { domain: [0, 1], anchor: "y2", matches: "x", showticklabels: false, gridcolor: "#edf1ef", zeroline: false },
    xaxis3: { domain: [0, 1], anchor: "y3", matches: "x", title: { text: "Time · seconds" }, gridcolor: "#edf1ef", zeroline: false },
    yaxis: { domain: [0.70, 1], title: { text: "F0 · st", font: { size: 9 } }, gridcolor: "#edf1ef", zeroline: false },
    yaxis2: { domain: [0.36, 0.64], title: { text: "Energy · dB", font: { size: 9 } }, gridcolor: "#edf1ef", zeroline: false },
    yaxis3: { domain: [0, 0.30], title: { text: "Rate · syl/s", font: { size: 9 } }, gridcolor: "#edf1ef", zeroline: false },
    legend: { orientation: "h", y: 1.12, x: 0, font: { size: 9 } }, hovermode: "x unified", shapes,
  }, { displayModeBar: false, responsive: true });
  timelineDraw.then(() => window.Plotly.Plots.resize("timeseries-chart"));
}

function updatePlotCursor(seconds) {
  if (!window.Plotly || $("#timeseries-chart").hidden || !state.currentResult) return;
  const existing = $("#timeseries-chart").layout?.shapes || [];
  const regions = visibleRegionEntries(state.currentResult).map(({ region }) => ({
    type: "rect", xref: "x", yref: "paper", x0: Number(region.start_s), x1: Number(region.end_s), y0: 0, y1: 1,
    fillcolor: regionColor(region.type), opacity: isReliableRegion(region) ? 0.10 : 0.035, line: { width: 0 }, layer: "below",
  }));
  regions.push({ type: "line", xref: "x", yref: "paper", x0: seconds, x1: seconds, y0: 0, y1: 1, line: { color: "#172d2d", width: 1.5, dash: "dot" } });
  if (existing.length) window.Plotly.relayout("timeseries-chart", { shapes: regions });
}

function renderReliability(metrics) {
  const target = $("#reliability-content");
  if (!metrics) {
    target.innerHTML = '<tr><td colspan="3" class="empty-state">Evaluation metrics are unavailable. No reliability values are being inferred.</td></tr>';
    $("#metrics-source").textContent = "Artifacts unavailable";
    return;
  }
  const pairedDev = metrics.scoring?.dev?.paired?.spearman_total_vs_severity;
  const pairedTest = metrics.scoring?.test?.paired?.spearman_total_vs_severity;
  const freeDev = metrics.scoring?.dev?.reference_free?.spearman_total_vs_severity;
  const freeTest = metrics.scoring?.test?.reference_free?.spearman_total_vs_severity;
  const pairedAuc = metrics.scoring?.test?.paired?.auc_ideal_vs_flawed;
  const pairedPerformance = metrics.test_performance?.paired || {};
  const gain = metrics.control_false_regions?.paired?.gain || {};
  const budget = metrics.false_region_budget_per_minute;
  const falseRate = pairedPerformance.false_regions_per_minute;
  const overBudget = Number.isFinite(falseRate) && Number.isFinite(budget) && falseRate > budget;
  const falseRateNote = `TEST controls + IDEAL; budget ${formatNumber(budget, 2)} per minute${overBudget ? " · slightly over budget" : ""}`;
  const weak = (metrics.weak_paired_detectors || []).map((type) => flawLabels[type] || type).join(", ") || "Unavailable";
  const unavailable = (metrics.missing_artifacts || []).length
    ? `<tr class="metrics-warning"><td colspan="3">Unavailable sources: ${escapeHtml(metrics.missing_artifacts.join(", "))}</td></tr>`
    : "";
  const blocks = [
    ["Paired DEV Spearman", pairedDev, "Score vs severity; paired ideal supplied", 2],
    ["Paired TEST Spearman", pairedTest, "Held-out passage severity correlation", 2],
    ["Reference-free DEV Spearman", freeDev, "Reference fitted from DEV IDEAL passages", 2],
    ["Reference-free TEST Spearman", freeTest, "Held-out passage severity correlation", 2],
    ["Paired TEST AUC", pairedAuc, "Ideal vs flawed; partly by construction because each read has its same-text ideal", 2],
    ["Paired false regions / min", falseRate, falseRateNote, 3],
    ["Paired boundary error", pairedPerformance.boundary_error_ms, "Mean TEST boundary error, milliseconds", 1],
    ["Gain-control false regions / min", gain.false_regions_per_minute, "Paired TEST controls; gain changes can cause false regions", 3],
    ["Weak paired detectors", weak, "TEST precision below 0.4", null],
  ];
  target.innerHTML = unavailable + blocks.map(([label, value, note, digits]) => {
    const display = typeof value === "number" && digits !== null
      ? formatNumber(value, digits)
      : value == null ? "unavailable" : escapeHtml(value);
    return `<tr><th scope="row">${escapeHtml(label)}</th><td>${display}</td><td>${escapeHtml(note)}</td></tr>`;
  }).join("");
  $("#metrics-source").textContent = metrics.missing_artifacts?.length ? "Partial evaluation artifacts" : "Evaluation results and frozen config";
  renderDetectorCoverage(metrics.detectors_by_mode || {});
}

function renderDetectorCoverage(modes) {
  const coverage = $("#detector-coverage");
  coverage.innerHTML = ["paired", "reference_free"].map((mode) => {
    const item = modes[mode] || { enabled: [], disabled: [] };
    const enabled = item.enabled?.length ? item.enabled.map((type) => flawLabels[type] || type).join(", ") : "Unavailable";
    const disabled = item.disabled?.length
      ? item.disabled.map((entry) => `<p><strong>${escapeHtml(flawLabels[entry.type] || entry.type)}:</strong> ${escapeHtml(entry.reason || "reason unavailable")}</p>`).join("")
      : "<p>None reported.</p>";
    return `<div class="detector-mode"><h3>${mode === "paired" ? "Paired" : "Reference-free"}</h3><p><strong>Enabled:</strong> ${escapeHtml(enabled)}</p><p><strong>Disabled:</strong></p>${disabled}</div>`;
  }).join("");
}

async function loadMetrics() {
  try {
    state.metrics = await getJson("/dashboard-metrics");
    renderReliability(state.metrics);
  } catch (_error) {
    state.metrics = null;
    renderReliability(null);
  }
}

function renderResult(result, { label = "", transcript = "", audioUrl = null, audioFile = null } = {}) {
  state.currentResult = result;
  state.currentId = label;
  state.transcript = transcript;
  state.showLowReliability = false;
  $("#mode-badge").textContent = scoreModeLabel(result.mode);
  $("#mode-badge").classList.toggle("paired", result.mode === "paired");
  $("#recording-label").textContent = label;
  $("#baseline-label").textContent = result.series?.baseline?.source || (result.mode === "paired" ? "Provided ideal" : "DEV-IDEAL reference");
  $("#summary-total-score").textContent = formatNumber(result.total_score, 1);
  renderSummary(result);
  updateLowToggle(result);
  renderDimensions(result);
  renderRadar(result);
  renderRegions(result);
  renderTranscript(result, transcript);
  state.duration = timelineDuration(result);
  renderWaveRegions(result);
  renderTimelines(result);
  setLoading(false);
  requestAnimationFrame(() => {
    if (window.Plotly) {
      window.Plotly.Plots.resize("radar-chart");
      window.Plotly.Plots.resize("timeseries-chart");
    }
  });
  setAudioSource(audioUrl, audioFile, label);
}

function setAudioSource(url, file, label) {
  state.waveformReady = false;
  if (state.wavesurfer) {
    state.wavesurfer.destroy();
    state.wavesurfer = null;
  }
  if (state.currentObjectUrl) {
    URL.revokeObjectURL(state.currentObjectUrl);
    state.currentObjectUrl = null;
  }
  const source = file ? (state.currentObjectUrl = URL.createObjectURL(file)) : url;
  const nativeAudio = $("#native-audio");
  state.currentAudio = nativeAudio;
  $("#audio-source-label").textContent = label || "Recording";
  if (!source) {
    nativeAudio.hidden = true;
    $("#waveform").classList.remove("is-loaded");
    return;
  }
  nativeAudio.src = source;
  nativeAudio.hidden = false;
  nativeAudio.onloadedmetadata = () => {
    if (Number.isFinite(nativeAudio.duration) && nativeAudio.duration > 0) {
      state.duration = nativeAudio.duration;
      renderWaveRegions(state.currentResult || { regions: [] });
    }
  };
  nativeAudio.onerror = () => {
    setBanner("The audio file is missing or could not be decoded. Scores and explanations are still available.");
    nativeAudio.hidden = true;
  };
  nativeAudio.ontimeupdate = () => {
    syncTranscript(nativeAudio.currentTime);
    updatePlotCursor(nativeAudio.currentTime);
  };
  if (!window.WaveSurfer) return;
  try {
    state.wavesurfer = window.WaveSurfer.create({
      container: "#waveform", height: 96, waveColor: "#94b4aa", progressColor: "#087f78",
      cursorColor: "#172d2d", cursorWidth: 2, barWidth: 2, barGap: 2, barRadius: 1,
      normalize: true, interact: true, media: nativeAudio,
    });
    state.wavesurfer.on("timeupdate", (seconds) => {
      syncTranscript(seconds);
      updatePlotCursor(seconds);
    });
    state.wavesurfer.on("ready", () => {
      requestAnimationFrame(() => {
        if (!$("#waveform").querySelector("canvas")) {
          state.wavesurfer?.destroy();
          state.wavesurfer = null;
          state.waveformReady = false;
          nativeAudio.hidden = false;
          $("#waveform").classList.remove("is-loaded");
          setBanner("Waveform rendering is unavailable. Use the audio player below; analysis remains available.");
          return;
        }
        state.waveformReady = true;
        nativeAudio.hidden = true;
        $("#waveform").classList.add("is-loaded");
        state.duration = state.wavesurfer.getDuration() || state.duration;
        renderWaveRegions(state.currentResult || { regions: [] });
      });
    });
    state.wavesurfer.on("error", () => {
      state.wavesurfer?.destroy();
      state.wavesurfer = null;
      state.waveformReady = false;
      $("#waveform").classList.remove("is-loaded");
      nativeAudio.hidden = false;
      setBanner("Waveform playback could not load. The browser audio player is available instead.");
    });
    state.wavesurfer.load(source);
  } catch (_error) {
    state.waveformReady = false;
    $("#waveform").classList.remove("is-loaded");
    nativeAudio.hidden = false;
    setBanner("Waveform playback is unavailable. The browser audio player is available instead.");
  }
}

async function loadDemo(recordingId) {
  if (!recordingId) return;
  setBanner("");
  $("#initial-state").hidden = true;
  $("#results").hidden = false;
  $("#loading-skeleton").hidden = false;
  $("#result-content").hidden = true;
  try {
    const result = await getJson(`/demo/${encodeURIComponent(recordingId)}`);
    const transcript = result.transcript || (result.series?.participant?.speech_rate_sps || []).map((item) => item.word).join(" ");
    renderResult(result, {
      label: result.recording_id || recordingId,
      transcript,
      audioUrl: `/audio/${encodeURIComponent(recordingId)}`,
    });
    if (!result.transcript && !result.series?.participant?.speech_rate_sps?.length) {
      setBanner("This precomputed result has no aligned word series; transcript synchronization is unavailable.");
    }
  } catch (error) {
    state.currentResult = null;
    setLoading(false);
    setBanner(error.message, "error");
  }
}

async function loadDemoIndex() {
  const selector = $("#demo-select");
  const empty = $("#demo-empty");
  try {
    const payload = await getJson("/demo");
    state.demoIds = Array.isArray(payload.recording_ids) ? payload.recording_ids : [];
    selector.replaceChildren();
    if (!state.demoIds.length) {
      selector.add(new Option("No demo recordings", ""));
      selector.disabled = true;
      $("#severity-slider").disabled = true;
      empty.hidden = false;
      return;
    }
    for (const id of state.demoIds) selector.add(new Option(prettyId(id), id));
    selector.disabled = false;
    empty.hidden = true;
    const idealId = state.demoIds.find((id) => /__ideal$/i.test(id)) || state.demoIds[0];
    selector.value = idealId;
    setLadderFor(idealId);
    await loadDemo(idealId);
  } catch (error) {
    selector.replaceChildren(new Option("Demo list unavailable", ""));
    selector.disabled = true;
    empty.hidden = false;
    empty.textContent = `Demo results unavailable: ${error.message}`;
  }
}

async function submitAnalysis(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const audioFile = $("#audio-file").files[0];
  const idealFile = $("#ideal-file").files[0];
  if (!audioFile) return;
  const data = new FormData();
  data.append("audio_file", audioFile);
  data.append("transcript", $("#transcript-input").value);
  if (idealFile) data.append("ideal_audio", idealFile);
  const button = $("#analyze-button");
  button.disabled = true;
  $("#analysis-progress").hidden = false;
  setBanner("");
  setLoading(true);
  try {
    const response = await fetch("/analyze", { method: "POST", body: data });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(responseMessage(payload, `Analysis failed (${response.status})`));
    renderResult(payload, {
      label: audioFile.name,
      transcript: $("#transcript-input").value,
      audioFile,
    });
    setTab("own");
  } catch (error) {
    state.currentResult = null;
    setLoading(false);
    setBanner(error.message, "error");
  } finally {
    button.disabled = false;
    $("#analysis-progress").hidden = true;
  }
}

function connectEvents() {
  $("#tab-demo").addEventListener("click", () => setTab("demo"));
  $("#tab-own").addEventListener("click", () => setTab("own"));
  $("#demo-select").addEventListener("change", (event) => {
    setLadderFor(event.target.value);
    loadDemo(event.target.value);
  });
  $("#severity-slider").addEventListener("input", (event) => {
    const level = Number(event.target.value);
    $("#severity-value").textContent = level === 0 ? "IDEAL" : `L${level}`;
  });
  $("#severity-slider").addEventListener("change", (event) => {
    const suffix = Number(event.target.value) === 0 ? "ideal" : `l${event.target.value}`;
    const id = state.demoByLadder[suffix];
    if (id) {
      $("#demo-select").value = id;
      loadDemo(id);
    }
  });
  $("#show-low-regions").addEventListener("change", (event) => {
    state.showLowReliability = event.currentTarget.checked;
    refreshFilteredRegions();
  });
  $("#analyze-form").addEventListener("submit", submitAnalysis);
  $("#play-button").addEventListener("click", () => {
    if (state.currentAudio) {
      if (state.currentAudio.paused) state.currentAudio.play().catch(() => {});
      else state.currentAudio.pause();
    }
  });
  window.addEventListener("resize", () => {
    if (window.Plotly) {
      if (!$("#radar-chart").hidden) window.Plotly.Plots.resize("radar-chart");
      if (!$("#timeseries-chart").hidden) window.Plotly.Plots.resize("timeseries-chart");
    }
  });
}

function showCdnNotice() {
  const failed = new Set(window.speechLensCdnFailures || []);
  if (!window.WaveSurfer) failed.add("WaveSurfer");
  if (!window.Plotly) failed.add("Plotly");
  if (failed.size) {
    const parts = [];
    if (failed.has("WaveSurfer")) parts.push("WaveSurfer failed to load; the standard audio player will be used.");
    if (failed.has("Plotly")) parts.push("Plotly failed to load; score bars, flaw table, transcript, and explanations remain available.");
    state.cdnNotice = parts.join(" ");
    setBanner(state.cdnNotice);
  }
}

async function initialize() {
  connectEvents();
  showCdnNotice();
  await Promise.all([loadDemoIndex(), loadMetrics()]);
}

initialize();
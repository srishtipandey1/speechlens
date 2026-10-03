"use strict";

const $ = (selector) => document.querySelector(selector);
const DEFAULT_DEMO_ID = "dev_f_1462__L3";
const dimensions = [
  ["pacing", "Pacing", "Word-duration similarity to the reference."],
  ["pausing_fluency", "Pausing / fluency", "Excess inter-word pauses."],
  ["intonation", "Intonation", "Pitch variation relative to the reference."],
  ["energy", "Energy", "Downward speech-level change."],
  ["articulation", "Articulation", "Syllable-nuclei rate and count agreement."],
  ["disfluency", "Disfluency", "Excess voiced time outside aligned words."],
];
const flawLabels = {
  pace_fast: "Fast pace", pace_slow: "Slow pace", long_pause: "Long pause",
  monotone: "Monotone", volume_dropoff: "Volume dropoff", filler: "Filler",
  stumble_repeat: "Stumble / repeat",
};
const flawColors = {
  pace_fast: "#cf5948", pace_slow: "#b57b1c", long_pause: "#4478a8",
  monotone: "#8d5d42", volume_dropoff: "#567d38", filler: "#9a5b8d",
  stumble_repeat: "#627889",
};
const state = {
  demoIds: [], demoByLadder: {}, currentId: "", result: null, transcript: "",
  audio: null, objectUrl: null, audioContext: null, audioBuffer: null, peaks: [],
  peakColumns: 0, duration: 0, lowVisible: false, frame: 0, dragging: false,
  scoreValue: 0, metrics: null, lastScrolledWord: "",
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function formatNumber(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "unavailable";
}

function formatTime(value) {
  const seconds = Math.max(0, Number(value) || 0);
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
}

function setStatus(panel, message, tone = "notice") {
  const target = $(`#${panel}-status`);
  if (!target) return;
  target.hidden = !message;
  target.className = `panel-status ${tone === "error" ? "is-error" : ""}`;
  target.querySelector("span").textContent = message || "";
}

function responseMessage(payload, fallback) {
  if (typeof payload?.detail === "string") return payload.detail;
  if (typeof payload?.detail?.message === "string") return payload.detail.message;
  return typeof payload?.message === "string" ? payload.message : fallback;
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

function ladderPrefix(recordingId) {
  return recordingId.replace(/__(?:ideal|L[1-5]|control_.+)$/i, "");
}

function setLadderFor(recordingId) {
  const prefix = ladderPrefix(recordingId);
  state.demoByLadder = {};
  for (const id of state.demoIds) {
    if (ladderPrefix(id) !== prefix) continue;
    const suffix = id.match(/__(ideal|L[1-5])$/i)?.[1]?.toLowerCase();
    if (suffix) state.demoByLadder[suffix] = id;
  }
  const suffix = recordingId.slice(prefix.length + 2).toLowerCase();
  const level = suffix === "ideal" ? 0 : /^l[1-5]$/.test(suffix) ? Number(suffix.slice(1)) : -1;
  $("#severity-slider").disabled = Object.keys(state.demoByLadder).length < 2;
  if (level >= 0) $("#severity-slider").value = String(level);
  $("#severity-value").textContent = level === 0 ? "Ideal" : level > 0 ? `L${level}` : "—";
}

function setLoading(loading) {
  $("#results").hidden = !loading && !state.result;
  $("#loading-skeleton").hidden = !loading;
  $("#result-content").hidden = loading || !state.result;
  $("#initial-state").hidden = loading || Boolean(state.result);
}

function reliable(region) {
  return ["high", "medium"].includes(region.reliability?.level);
}

function shownRegions(result = state.result) {
  return (result?.regions || []).map((region, index) => ({ region, index }))
    .filter(({ region }) => reliable(region) || state.lowVisible);
}

function scoreBand(value) {
  if (value >= 80) return ["Close to reference", "close"];
  if (value >= 60) return ["Noticeable differences", "noticeable"];
  return ["Substantial differences", "substantial"];
}

function summarySentence(region) {
  return String(region.sentence || "Timing differs from the reference here.")
    .replace(/^\d{2}:\d{2}\.\d-\d{2}:\d{2}\.\d\s*/, "");
}

function deviation(region) {
  const observed = Number(region.observed_numeric);
  const expected = Number(region.expected_numeric);
  if (Number.isFinite(observed) && Number.isFinite(expected)) return Math.abs(observed - expected);
  const ratio = Number(region.z_score_or_ratio);
  return Number.isFinite(ratio) ? Math.abs(ratio) : Number(region.severity) || 0;
}

function renderSummary(result) {
  const total = Number(result.total_score ?? 0);
  $("#summary-total-score").textContent = formatNumber(total, 1);
  const [label, className] = scoreBand(total);
  $("#score-band").textContent = label;
  $("#score-band").className = `score-band ${className}`;
  const findings = (result.regions || []).map((region, index) => ({ region, index }))
    .sort((a, b) => ({ high: 0, medium: 1, low: 2 }[a.region.reliability?.level] ?? 3)
      - ({ high: 0, medium: 1, low: 2 }[b.region.reliability?.level] ?? 3)
      || deviation(b.region) - deviation(a.region)
      || Number(a.region.start_s) - Number(b.region.start_s))
    .slice(0, 3);
  const list = $("#summary-findings-list");
  if (!findings.length) {
    list.innerHTML = '<li class="empty-state">No flaw regions were detected.</li>';
    return;
  }
  list.innerHTML = findings.map(({ region, index }) => `<li>
    <button class="finding-link" type="button" data-region-index="${index}">
      <span class="finding-time">${formatTime(region.start_s)}</span>
      <span class="finding-copy"><strong>${escapeHtml(summarySentence(region))}</strong><small>${escapeHtml(region.suggestion || "Review this delivery segment.")}</small></span>
      <span class="finding-reliability ${escapeHtml(region.reliability?.level || "unavailable")}">${escapeHtml(region.reliability?.level || "unavailable")}</span>
    </button></li>`).join("");
  list.querySelectorAll("button[data-region-index]").forEach((button) => {
    button.addEventListener("click", () => {
      const index = Number(button.dataset.regionIndex);
      const region = result.regions[index];
      if (!reliable(region) && !state.lowVisible) {
        state.lowVisible = true;
        refreshFilteredRegions();
      }
      seekTo(region.start_s, index);
    });
  });
}

function animateScore(target) {
  const start = state.scoreValue;
  const started = performance.now();
  function tick(now) {
    const progress = Math.min(1, (now - started) / 420);
    $("#summary-total-score").textContent = (start + (target - start) * (1 - (1 - progress) ** 3)).toFixed(1);
    if (progress < 1) requestAnimationFrame(tick);
    else state.scoreValue = target;
  }
  requestAnimationFrame(tick);
}

function renderDimensions(result) {
  const root = $("#dimension-list");
  root.replaceChildren();
  for (const [key, label, note] of dimensions) {
    const score = Number(result.scores?.[key] ?? 0);
    const row = document.createElement("div");
    row.className = "dimension-row";
    row.innerHTML = `<div class="dimension-label"><strong>${escapeHtml(label)}</strong><small>${escapeHtml(note)}</small></div>
      <div class="dimension-track" role="meter" aria-label="${escapeHtml(label)} score" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${score}"><span class="dimension-fill"></span></div>
      <span class="dimension-value">${formatNumber(score, 0)}</span>`;
    root.append(row);
    requestAnimationFrame(() => { row.querySelector(".dimension-fill").style.width = `${Math.max(0, Math.min(100, score))}%`; });
  }
  animateScore(Number(result.total_score ?? 0));
  const referenceRead = result.kind === "ideal" || /__ideal$/i.test(state.currentId);
  $("#reference-note").textContent = referenceRead
    ? "This is the reference reading; participant and baseline are identical."
    : result.mode === "reference_free"
      ? "Reference-free scores use DEV-IDEAL statistics. Pausing/fluency and disfluency have enabled detector support; other dimensions are limited."
      : "Paired mode compares this read with its same-text ideal.";
  $("#reference-timeline-note").hidden = !referenceRead;
}

function renderRadar(result) {
  if (!window.Plotly) {
    $("#radar-chart").hidden = true;
    $("#radar-fallback").hidden = false;
    return;
  }
  $("#radar-chart").hidden = false;
  $("#radar-fallback").hidden = true;
  const labels = dimensions.map(([, label]) => label);
  const scores = dimensions.map(([key]) => Number(result.scores?.[key] ?? 0));
  const draw = window.Plotly.react("radar-chart", [{
    type: "scatterpolar",
    r: [...scores, scores[0]],
    theta: [...labels, labels[0]],
    fill: "toself",
    fillcolor: "rgba(36,110,97,.14)",
    line: { color: "#246e61", width: 2 },
    marker: { color: "#246e61", size: 4 },
    hovertemplate: "%{theta}: %{r:.1f}<extra></extra>",
  }], {
    margin: { t: 12, r: 30, b: 24, l: 30 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    showlegend: false,
    polar: {
      bgcolor: "transparent",
      radialaxis: { range: [0, 100], tickvals: [0, 50, 100], tickfont: { size: 9, color: "#5e6a67" }, gridcolor: "#d8ddda" },
      angularaxis: { tickfont: { size: 9, color: "#26312f" }, gridcolor: "#d8ddda" },
    },
  }, { displayModeBar: false, responsive: true });
  draw.then(() => window.Plotly.Plots.resize("radar-chart"));
}

function updateLadder(result) {
  const total = result.regions?.length || 0;
  const low = (result.regions || []).filter((region) => !reliable(region)).length;
  $("#show-low-regions").checked = state.lowVisible;
  $("#show-low-regions").disabled = low === 0;
  $("#low-toggle-label").textContent = `Show low-reliability regions (${low})`;
  $("#region-count").textContent = String(shownRegions(result).length);
  return total;
}

function regionColor(type) {
  return flawColors[type] || "#65788a";
}

function rgba(hex, alpha) {
  const values = hex.match(/[0-9a-f]{2}/gi)?.map((part) => parseInt(part, 16)) || [80, 100, 90];
  return `rgba(${values[0]},${values[1]},${values[2]},${alpha})`;
}

function peaksForBuffer(buffer, columns) {
  const peaks = [];
  for (let column = 0; column < columns; column += 1) {
    const start = Math.floor(column * buffer.length / columns);
    const end = Math.max(start + 1, Math.floor((column + 1) * buffer.length / columns));
    let minimum = 1;
    let maximum = -1;
    for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
      const samples = buffer.getChannelData(channel);
      for (let index = start; index < end && index < buffer.length; index += 1) {
        minimum = Math.min(minimum, samples[index]);
        maximum = Math.max(maximum, samples[index]);
      }
    }
    peaks.push({ minimum, maximum });
  }
  return peaks;
}

function drawWaveform() {
  const canvas = $("#waveform-canvas");
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const pixelRatio = Math.max(1, window.devicePixelRatio || 1);
  const columns = Math.max(1, Math.floor(bounds.width));
  const backingWidth = Math.round(bounds.width * pixelRatio);
  const backingHeight = Math.round(bounds.height * pixelRatio);
  if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
    canvas.width = backingWidth;
    canvas.height = backingHeight;
  }
  if (state.audioBuffer && state.peaks.length !== columns) state.peaks = peaksForBuffer(state.audioBuffer, columns);
  state.peakColumns = columns;
  canvas.dataset.peakColumns = String(state.peaks.length);
  canvas.dataset.regionOverlays = String(shownRegions().length);
  const context = canvas.getContext("2d");
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, bounds.width, bounds.height);
  context.fillStyle = "#fff";
  context.fillRect(0, 0, bounds.width, bounds.height);
  const duration = Math.max(state.duration, 0.001);
  for (const { region } of shownRegions()) {
    const left = Math.max(0, Number(region.start_s) / duration * bounds.width);
    const right = Math.min(bounds.width, Number(region.end_s) / duration * bounds.width);
    context.fillStyle = rgba(regionColor(region.type), reliable(region) ? 0.15 : 0.045);
    context.fillRect(left, 0, Math.max(1, right - left), bounds.height);
  }
  const middle = bounds.height / 2;
  const amplitude = bounds.height * 0.45;
  context.beginPath();
  context.strokeStyle = "#65736e";
  context.lineWidth = 1;
  state.peaks.forEach((peak, column) => {
    context.moveTo(column + 0.5, middle - peak.maximum * amplitude);
    context.lineTo(column + 0.5, middle - peak.minimum * amplitude);
  });
  context.stroke();
  const cursor = Math.max(0, Math.min(bounds.width, (state.audio?.currentTime || 0) / duration * bounds.width));
  context.beginPath();
  context.strokeStyle = "#246e61";
  context.lineWidth = 1.5;
  context.moveTo(cursor + 0.5, 0);
  context.lineTo(cursor + 0.5, bounds.height);
  context.stroke();
}

function renderWaveRegions(result) {
  const overlay = $("#wave-regions");
  overlay.replaceChildren();
  for (const { region, index } of shownRegions(result)) {
    const start = Number(region.start_s) || 0;
    const end = Number(region.end_s) || start;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "wave-region";
    button.style.left = `${Math.max(0, start / Math.max(state.duration, 0.001) * 100)}%`;
    button.style.width = `${Math.max(1.2, (end - start) / Math.max(state.duration, 0.001) * 100)}%`;
    button.style.setProperty("--type-color", regionColor(region.type));
    button.setAttribute("aria-label", `Seek to ${flawLabels[region.type] || region.type} at ${formatTime(start)} (${region.reliability?.level || "unavailable"} reliability)`);
    button.title = `${formatTime(start)}–${formatTime(end)} · ${flawLabels[region.type] || region.type}`;
    button.addEventListener("click", () => seekTo(start, index));
    overlay.append(button);
  }
  drawWaveform();
}

function renderTranscript(result, text = "") {
  const root = $("#transcript-words");
  state.lastScrolledWord = "";
  const words = result.series?.participant?.speech_rate_sps || text.trim().split(/\s+/).filter(Boolean).map((word) => ({ word }));
  if (!words.length) {
    root.innerHTML = '<p class="empty-state">No aligned word timings were returned.</p>';
    return;
  }
  root.replaceChildren();
  words.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "word-chip";
    button.textContent = item.word;
    button.dataset.start = String(item.start_s ?? "");
    button.dataset.end = String(item.end_s ?? "");
    const start = Number(item.start_s);
    const end = Number(item.end_s);
    if (shownRegions(result).some(({ region }) => start < Number(region.end_s) && end > Number(region.start_s))) {
      button.classList.add("is-flawed");
    }
    button.setAttribute("aria-label", Number.isFinite(start) ? `Seek to ${item.word} at ${formatTime(start)}` : item.word);
    button.addEventListener("click", () => { if (Number.isFinite(start)) seekTo(start); });
    root.append(button);
    if (index < words.length - 1) root.append(document.createTextNode(" "));
  });
}

function syncTranscript(seconds, autoScroll = false) {
  let active = null;
  document.querySelectorAll(".word-chip").forEach((word) => {
    const start = Number(word.dataset.start);
    const end = Number(word.dataset.end);
    const selected = Number.isFinite(start) && seconds >= start && seconds <= end;
    word.classList.toggle("is-current", selected);
    if (selected) active = word;
  });
  if (autoScroll && active && active.dataset.start !== state.lastScrolledWord) {
    active.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
    state.lastScrolledWord = active.dataset.start;
  }
  $("#playback-time").textContent = formatTime(seconds);
}

function seekTo(seconds, selectedIndex = -1) {
  if (!state.audio) return;
  document.querySelectorAll("#region-rows tr[data-region-index]").forEach((row) => {
    row.classList.toggle("is-selected", Number(row.dataset.regionIndex) === selectedIndex);
  });
  state.audio.currentTime = Math.max(0, Math.min(state.duration, Number(seconds) || 0));
  state.audio.play().catch(() => {});
  syncTranscript(state.audio.currentTime, true);
  updatePlotCursor(state.audio.currentTime);
  drawWaveform();
}

function seekCanvas(event) {
  if (!state.audio || state.duration <= 0) return;
  const bounds = $("#waveform-canvas").getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width));
  state.audio.currentTime = ratio * state.duration;
  syncTranscript(state.audio.currentTime, true);
  updatePlotCursor(state.audio.currentTime);
  drawWaveform();
}

async function decodeWaveform(source) {
  const response = await fetch(source);
  if (!response.ok) throw new Error(`Audio request failed (${response.status})`);
  const encoded = await response.arrayBuffer();
  const AudioContextType = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextType) throw new Error("Web Audio decoding is not supported in this browser");
  if (!state.audioContext || state.audioContext.state === "closed") state.audioContext = new AudioContextType();
  state.audioBuffer = await state.audioContext.decodeAudioData(encoded.slice(0));
  state.duration = state.audioBuffer.duration;
  state.peaks = peaksForBuffer(state.audioBuffer, Math.max(1, Math.floor($("#waveform-canvas").clientWidth)));
  drawWaveform();
  $("#waveform").classList.add("is-loaded");
}

function animatePlayback() {
  if (!state.audio) return;
  const seconds = state.audio.currentTime;
  syncTranscript(seconds, true);
  updatePlotCursor(seconds);
  drawWaveform();
  if (!state.audio.paused && !state.audio.ended) state.frame = requestAnimationFrame(animatePlayback);
  else state.frame = 0;
}

function renderRegions(result) {
  const root = $("#region-rows");
  const entries = shownRegions(result);
  $("#region-count").textContent = String(entries.length);
  if (!entries.length) {
    root.innerHTML = '<tr><td colspan="5" class="empty-state">No high- or medium-reliability regions are shown.</td></tr>';
    return;
  }
  root.innerHTML = entries.map(({ region, index }) => {
    const type = region.type || "other_deviation";
    const reliability = region.reliability || {};
    const measured = [
      region.observed_value || "Observed value unavailable",
      `Expected: ${region.expected_value || "unavailable"}`,
      `Ratio / z: ${formatNumber(region.z_score_or_ratio, 2)}`,
      `Formula: ${region.formula || "unavailable"}`,
    ].join(" · ");
    const explanation = [region.sentence, region.suggestion].filter(Boolean).join(" ");
    return `<tr data-region-index="${index}">
      <td><button class="time-link" type="button" data-seek="${Number(region.start_s) || 0}" data-region="${index}">${formatTime(region.start_s)}–${formatTime(region.end_s)}</button></td>
      <td><span class="type-tag" style="--type-color:${regionColor(type)}">${escapeHtml(flawLabels[type] || type)}</span></td>
      <td class="measured-cell">${escapeHtml(measured)}</td>
      <td class="explanation-cell">${region.words?.length ? `<strong>${escapeHtml(region.words.join(" "))}</strong>` : ""}<span>${escapeHtml(explanation || "Explanation unavailable")}</span></td>
      <td><span class="reliability-badge ${escapeHtml(reliability.level || "unavailable")}" title="${escapeHtml(reliabilityTooltip(reliability))}">${escapeHtml(reliability.level || "unavailable")}</span></td>
    </tr>`;
  }).join("");
  root.querySelectorAll("button[data-seek]").forEach((button) => {
    button.addEventListener("click", () => seekTo(Number(button.dataset.seek), Number(button.dataset.region)));
  });
}

function reliabilityTooltip(reliability) {
  if (reliability.test_precision == null || reliability.test_recall == null) return "Held-out test precision and recall unavailable";
  return `Held-out TEST precision: ${Number(reliability.test_precision).toFixed(6)}; recall: ${Number(reliability.test_recall).toFixed(6)}; IoU: ${Number(reliability.iou_threshold ?? 0.3).toFixed(1)}`;
}

function refreshRegions() {
  if (!state.result) return;
  updateLadder(state.result);
  renderRegions(state.result);
  renderTranscript(state.result, state.transcript);
  renderWaveRegions(state.result);
  renderTimelines(state.result);
}

function median(values) {
  if (!values.length) return null;
  const middle = Math.floor(values.length / 2);
  return values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

function smoothedSeries(points, windowSeconds, interval = false, clipMaximum = null) {
  const times = points.map((point) => interval
    ? (Number(point.start_s) + Number(point.end_s)) / 2
    : Number(point.time_s));
  const values = points.map((point) => point.value == null ? null : Number(point.value));
  const radius = windowSeconds / 2;
  let first = 0;
  let afterLast = 0;
  const output = [];
  for (let index = 0; index < points.length; index += 1) {
    while (first < index && times[first] < times[index] - radius) first += 1;
    afterLast = Math.max(afterLast, index + 1);
    while (afterLast < points.length && times[afterLast] <= times[index] + radius) afterLast += 1;
    if (values[index] == null || !Number.isFinite(values[index])) {
      output.push(null);
      continue;
    }
    const window = values.slice(first, afterLast).filter((value) => value != null && Number.isFinite(value)).sort((a, b) => a - b);
    let value = median(window);
    if (clipMaximum != null) value = Math.max(0, Math.min(clipMaximum, value));
    output.push(value);
  }
  return { x: times, y: output };
}

function renderTimelines(result) {
  if (!window.Plotly) {
    $("#timeseries-chart").hidden = true;
    $("#timeseries-fallback").hidden = false;
    setStatus("timeline", "Local Plotly could not load. The score and region details remain available.");
    return;
  }
  $("#timeseries-chart").hidden = false;
  $("#timeseries-fallback").hidden = true;
  setStatus("timeline", "");
  const participant = result.series?.participant || {};
  const baseline = result.series?.baseline || {};
  const measures = [
    ["F0 · semitones", participant.f0_semitones, baseline.f0_semitones, "y", "x", "#246e61", 0.08, false, null],
    ["Energy · dB", participant.energy_db, baseline.energy_db, "y2", "x2", "#cf5948", 0.25, false, null],
    ["Speech rate · syllables/s", participant.speech_rate_sps, baseline.speech_rate_sps, "y3", "x3", "#4478a8", 1.5, true, 12],
  ];
  const traces = [];
  for (const [name, actual, reference, yaxis, xaxis, color, windowSeconds, interval, clipMaximum] of measures) {
    const actualLine = smoothedSeries(actual || [], windowSeconds, interval, clipMaximum);
    traces.push({ ...actualLine, type: "scatter", mode: "lines", name: `Participant · ${name}`, xaxis, yaxis, line: { color, width: 1.8 }, connectgaps: false });
    if (reference?.length) {
      const referenceLine = smoothedSeries(reference, windowSeconds, interval, clipMaximum);
      traces.push({ ...referenceLine, type: "scatter", mode: "lines", name: `Baseline · ${name}`, xaxis, yaxis, line: { color, width: 1.4, dash: "dot" }, connectgaps: false });
    }
  }
  const shapes = shownRegions(result).map(({ region }) => ({
    type: "rect", xref: "x", yref: "paper", x0: Number(region.start_s), x1: Number(region.end_s), y0: 0, y1: 1,
    fillcolor: regionColor(region.type), opacity: reliable(region) ? 0.1 : 0.035, line: { width: 0 }, layer: "below",
  }));
  shapes.push({ type: "line", xref: "x", yref: "paper", x0: state.audio?.currentTime || 0, x1: state.audio?.currentTime || 0, y0: 0, y1: 1, line: { color: "#26312f", width: 1.5, dash: "dot" } });
  const draw = window.Plotly.react("timeseries-chart", traces, {
    height: 400, margin: { l: 62, r: 20, t: 38, b: 35 }, paper_bgcolor: "transparent", plot_bgcolor: "#ffffff",
    font: { family: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 10, color: "#5e6a67" },
    grid: { rows: 3, columns: 1, pattern: "independent" },
    xaxis: { domain: [0, 1], anchor: "y", range: [0, state.duration], showticklabels: false, gridcolor: "#edf1ef", zeroline: false },
    xaxis2: { domain: [0, 1], anchor: "y2", matches: "x", showticklabels: false, gridcolor: "#edf1ef", zeroline: false },
    xaxis3: { domain: [0, 1], anchor: "y3", matches: "x", title: { text: "Time · seconds" }, gridcolor: "#edf1ef", zeroline: false },
    yaxis: { domain: [0.7, 1], title: { text: "F0 · st", font: { size: 9 } }, gridcolor: "#edf1ef", zeroline: false },
    yaxis2: { domain: [0.36, 0.64], title: { text: "Energy · dB", font: { size: 9 } }, gridcolor: "#edf1ef", zeroline: false },
    yaxis3: { domain: [0, 0.3], range: [0, 12], title: { text: "Speech rate · 0–12 syllables/s (clipped)", font: { size: 9 } }, gridcolor: "#edf1ef", zeroline: false },
    legend: { orientation: "h", y: 1.12, x: 0, font: { size: 9 } }, hovermode: "x unified", shapes,
  }, { displayModeBar: false, responsive: true });
  draw.then(() => window.Plotly.Plots.resize("timeseries-chart"));
}

function updatePlotCursor(seconds) {
  if (!window.Plotly || $("#timeseries-chart").hidden || !state.result) return;
  const shapes = shownRegions(state.result).map(({ region }) => ({
    type: "rect", xref: "x", yref: "paper", x0: Number(region.start_s), x1: Number(region.end_s), y0: 0, y1: 1,
    fillcolor: regionColor(region.type), opacity: reliable(region) ? 0.1 : 0.035, line: { width: 0 }, layer: "below",
  }));
  shapes.push({ type: "line", xref: "x", yref: "paper", x0: seconds, x1: seconds, y0: 0, y1: 1, line: { color: "#26312f", width: 1.5, dash: "dot" } });
  window.Plotly.relayout("timeseries-chart", { shapes });
}

function renderReliability(metrics) {
  const body = $("#reliability-content");
  if (!metrics) {
    body.innerHTML = '<tr><td colspan="3" class="empty-state">Evaluation metrics unavailable.</td></tr>';
    $("#metrics-source").textContent = "Artifacts unavailable";
    return;
  }
  const paired = metrics.test_performance?.paired || {};
  const gain = metrics.control_false_regions?.paired?.gain || {};
  const budget = metrics.false_region_budget_per_minute;
  const rate = paired.false_regions_per_minute;
  const overBudget = Number.isFinite(rate) && Number.isFinite(budget) && rate > budget;
  const weak = (metrics.weak_paired_detectors || []).join(", ") || "None reported";
  const rows = [
    ["Paired DEV Spearman", metrics.scoring?.dev?.paired?.spearman_total_vs_severity, "Score vs severity; paired ideal supplied", 2],
    ["Paired TEST Spearman", metrics.scoring?.test?.paired?.spearman_total_vs_severity, "Held-out passage severity correlation", 2],
    ["Reference-free DEV Spearman", metrics.scoring?.dev?.reference_free?.spearman_total_vs_severity, "Reference fitted from DEV IDEAL passages", 2],
    ["Reference-free TEST Spearman", metrics.scoring?.test?.reference_free?.spearman_total_vs_severity, "Held-out passage severity correlation", 2],
    ["Paired TEST AUC", metrics.scoring?.test?.paired?.auc_ideal_vs_flawed, "Ideal vs flawed; partly by construction because the same-text ideal is supplied", 2],
    ["Paired TEST false regions/min", rate, `Budget ${formatNumber(budget, 2)} per minute${overBudget ? "; slightly over budget" : ""}`, 2],
    ["Paired TEST boundary error", paired.boundary_error_ms, "Mean boundary error, milliseconds", 1],
    ["Gain-control false regions/min", gain.false_regions_per_minute, "Paired TEST; gain changes can cause false regions", 2],
    ["Weak paired detectors", weak, "TEST precision below the reliability threshold", null],
  ];
  body.innerHTML = `${(metrics.missing_artifacts || []).length ? `<tr><td colspan="3">Unavailable: ${escapeHtml(metrics.missing_artifacts.join(", "))}</td></tr>` : ""}${rows.map(([label, value, note, digits]) => {
    const display = typeof value === "number" && digits !== null
      ? formatNumber(value, digits)
      : value == null ? "unavailable" : escapeHtml(value);
    return `<tr><th scope="row">${escapeHtml(label)}</th><td>${display}</td><td>${escapeHtml(note)}</td></tr>`;
  }).join("")}`;
  $("#metrics-source").textContent = metrics.missing_artifacts?.length ? "Partial evaluation results" : "Evaluation results";
  renderDetectorCoverage(metrics.detectors_by_mode || {});
}

function timelineDuration(result) {
  const participant = result.series?.participant || {};
  const timestamps = [
    ...(participant.f0_semitones || []).map((point) => Number(point.time_s) || 0),
    ...(participant.energy_db || []).map((point) => Number(point.time_s) || 0),
    ...(participant.speech_rate_sps || []).map((point) => Number(point.end_s) || 0),
    ...(result.regions || []).map((region) => Number(region.end_s) || 0),
  ];
  return Math.max(0.1, ...timestamps);
}

function renderDetectorCoverage(modes) {
  $("#detector-coverage").innerHTML = ["paired", "reference_free"].map((mode) => {
    const item = modes[mode] || { enabled: [], disabled: [] };
    const enabled = item.enabled?.length ? item.enabled.join(", ") : "unavailable";
    const disabled = item.disabled?.length
      ? item.disabled.map((entry) => `<p><strong>${escapeHtml(entry.type)}:</strong> ${escapeHtml(entry.reason)}</p>`).join("")
      : "<p>None reported.</p>";
    return `<div class="detector-mode"><h3>${mode === "paired" ? "Paired" : "Reference-free"}</h3><p><strong>Enabled:</strong> ${escapeHtml(enabled)}</p><p><strong>Disabled:</strong></p>${disabled}</div>`;
  }).join("");
}

async function loadMetrics() {
  try { state.metrics = await getJson("/dashboard-metrics"); renderReliability(state.metrics); }
  catch (_error) { renderReliability(null); }
}

function renderResult(result, { label = "", transcript = "", audioUrl = null, audioFile = null } = {}) {
  state.result = result;
  state.currentId = label;
  state.transcript = transcript;
  state.lowVisible = false;
  $("#mode-badge").textContent = result.mode === "paired" ? "Paired" : "Reference-free";
  $("#recording-label").textContent = label;
  $("#baseline-label").textContent = result.series?.baseline?.source || (result.mode === "paired" ? "Provided ideal" : "DEV-IDEAL reference");
  renderSummary(result);
  updateLadder(result);
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
    drawWaveform();
  });
  void setAudioSource(audioUrl, audioFile, label);
}

async function setAudioSource(url, file, label) {
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  if (state.audio) state.audio.pause();
  state.audioBuffer = null;
  state.peaks = [];
  state.peakColumns = 0;
  $("#waveform").classList.remove("is-loaded");
  setStatus("wave", "");
  const source = file ? (state.objectUrl = URL.createObjectURL(file)) : url;
  const audio = $("#native-audio");
  state.audio = audio;
  $("#audio-source-label").textContent = label || "Recording";
  if (!source) {
    audio.hidden = true;
    drawWaveform();
    return;
  }
  audio.hidden = false;
  audio.src = source;
  audio.load();
  audio.onloadedmetadata = () => {
    if (Number.isFinite(audio.duration) && audio.duration > 0) state.duration = audio.duration;
    renderWaveRegions(state.result || {});
    drawWaveform();
  };
  audio.ontimeupdate = () => {
    syncTranscript(audio.currentTime, !audio.paused);
    updatePlotCursor(audio.currentTime);
    drawWaveform();
  };
  audio.onplay = () => {
    if (state.frame) cancelAnimationFrame(state.frame);
    state.frame = requestAnimationFrame(animatePlayback);
  };
  audio.onpause = () => {
    if (state.frame) cancelAnimationFrame(state.frame);
    state.frame = 0;
    animatePlayback();
  };
  audio.onerror = () => setStatus("wave", "Audio playback could not load. Scores and explanations remain available.", "error");
  try {
    const response = await fetch(source);
    if (!response.ok) throw new Error(`Audio request failed (${response.status})`);
    const encoded = await response.arrayBuffer();
    const Context = window.AudioContext || window.webkitAudioContext;
    if (!Context) throw new Error("Web Audio decoding is not supported in this browser");
    if (!state.audioContext || state.audioContext.state === "closed") state.audioContext = new Context();
    state.audioBuffer = await state.audioContext.decodeAudioData(encoded.slice(0));
    state.duration = state.audioBuffer.duration;
    state.peaks = peaksForBuffer(state.audioBuffer, Math.max(1, Math.floor($("#waveform-canvas").clientWidth)));
    $("#waveform").classList.add("is-loaded");
    setStatus("wave", "");
    drawWaveform();
  } catch (error) {
    state.audioBuffer = null;
    state.peaks = [];
    drawWaveform();
    setStatus("wave", `Waveform decoding failed: ${error.message}. Audio playback and analysis remain available.`, "error");
  }
}

async function loadDemo(recordingId) {
  if (!recordingId) return;
  setStatus("demo", "");
  $("#initial-state").hidden = true;
  $("#results").hidden = false;
  $("#loading-skeleton").hidden = false;
  $("#result-content").hidden = true;
  try {
    const result = await getJson(`/demo/${encodeURIComponent(recordingId)}`);
    const transcript = result.transcript || (result.series?.participant?.speech_rate_sps || []).map((item) => item.word).join(" ");
    renderResult(result, { label: result.recording_id || recordingId, transcript, audioUrl: `/audio/${encodeURIComponent(recordingId)}` });
  } catch (error) {
    state.result = null;
    setLoading(false);
    setStatus("demo", error.message, "error");
  }
}

async function loadDemoIndex() {
  const selector = $("#demo-select");
  try {
    const payload = await getJson("/demo");
    state.demoIds = Array.isArray(payload.recording_ids) ? payload.recording_ids : [];
    selector.replaceChildren();
    if (!state.demoIds.length) {
      selector.add(new Option("No demo recordings", ""));
      selector.disabled = true;
      $("#severity-slider").disabled = true;
      $("#demo-empty").hidden = false;
      return;
    }
    state.demoIds.forEach((id) => selector.add(new Option(id, id)));
    selector.disabled = false;
    const defaultId = state.demoIds.includes(DEFAULT_DEMO_ID)
      ? DEFAULT_DEMO_ID
      : state.demoIds.find((id) => /__ideal$/i.test(id)) || state.demoIds[0];
    selector.value = defaultId;
    setLadderFor(defaultId);
    await loadDemo(defaultId);
  } catch (error) {
    selector.replaceChildren(new Option("Demo list unavailable", ""));
    selector.disabled = true;
    $("#demo-empty").hidden = false;
    setStatus("demo", error.message, "error");
  }
}

async function submitAnalysis(event) {
  event.preventDefault();
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
  setStatus("own", "");
  setLoading(true);
  try {
    const response = await fetch("/analyze", { method: "POST", body: data });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(responseMessage(payload, `Analysis failed (${response.status})`));
    renderResult(payload, { label: audioFile.name, transcript: $("#transcript-input").value, audioFile });
    setTab("own");
  } catch (error) {
    state.result = null;
    setLoading(false);
    setStatus("own", error.message, "error");
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
    $("#severity-value").textContent = level === 0 ? "Ideal" : `L${level}`;
  });
  $("#severity-slider").addEventListener("change", (event) => {
    const suffix = Number(event.target.value) === 0 ? "ideal" : `l${event.target.value}`;
    const id = state.demoByLadder[suffix];
    if (id) { $("#demo-select").value = id; loadDemo(id); }
  });
  $("#show-low-regions").addEventListener("change", (event) => {
    state.lowVisible = event.currentTarget.checked;
    refreshRegions();
  });
  $("#analyze-form").addEventListener("submit", submitAnalysis);
  $("#play-button").addEventListener("click", () => {
    if (!state.audio) return;
    if (state.audio.paused) state.audio.play().catch(() => {});
    else state.audio.pause();
  });
  const canvas = $("#waveform-canvas");
  canvas.addEventListener("pointerdown", (event) => {
    state.dragging = true;
    canvas.setPointerCapture(event.pointerId);
    seekCanvas(event);
  });
  canvas.addEventListener("pointermove", (event) => { if (state.dragging) seekCanvas(event); });
  canvas.addEventListener("pointerup", () => { state.dragging = false; });
  canvas.addEventListener("pointercancel", () => { state.dragging = false; });
  canvas.addEventListener("keydown", (event) => {
    if (!state.audio || !["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    state.audio.currentTime = Math.max(0, state.audio.currentTime + (event.key === "ArrowLeft" ? -1 : 1));
  });
  document.querySelectorAll(".panel-status button").forEach((button) => {
    button.addEventListener("click", () => { button.closest(".panel-status").hidden = true; });
  });
  window.addEventListener("resize", () => {
    if (window.Plotly) {
      if (!$("#radar-chart").hidden) window.Plotly.Plots.resize("radar-chart");
      if (!$("#timeseries-chart").hidden) window.Plotly.Plots.resize("timeseries-chart");
    }
    drawWaveform();
  });
}

async function initialize() {
  connectEvents();
  await Promise.all([loadDemoIndex(), loadMetrics()]);
}

initialize();
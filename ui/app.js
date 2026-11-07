"use strict";

/* ---------- Settings ---------- */

// Set to true to use fixed sample responses instead of the real API.
const USE_MOCK = false;

const API_URL = "/api/predict";
const EXAMPLES_URL = "examples/examples.json";
const MOCK_RESPONSE_URL = "mock/predict_response.json";

// The 0.5 threshold from the study's analysis. The model is calibrated to the training mix (17% patients).
const SCREEN_CUTOFF = 0.5;

const MAX_UPLOAD_MB = 4;
const MAX_FEATURES = 5;
const SLOW_NOTICE_MS = 8000;
const REQUEST_TIMEOUT_MS = 30000;
const MOCK_DELAY_MS = 900;
const COUNTDOWN_SECONDS = 3;
const RECORD_SECONDS = 2.5;
const SILENCE_PEAK = 0.01;

// Mock mode only: add ?mock=slow or ?mock=error to the URL to preview those states.
const MOCK_SCENARIO = USE_MOCK ? new URLSearchParams(location.search).get("mock") : null;

/* ---------- App state ---------- */

const state = {
  mode: "examples", // "examples" or "recording"
  examples: null,
  examplesError: null,
  selectedExample: null,
  audio: null, // { blob, name, autoScore }
  // status: "idle", "loading", "error" or "done"
  prediction: { status: "idle" },
};

let activeRequest = null;
let activeRecording = null;
let audioUrl = null;

const $ = (id) => document.getElementById(id);
const resultEl = $("result");

/** Creates an element. Text goes in through text nodes, so API strings can't inject HTML. */
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "style") node.style.cssText = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  node.append(...children.filter((child) => child !== null && child !== undefined));
  return node;
}

/* ---------- Validation and formatting ---------- */

function parseFeatures(raw) {
  if (!Array.isArray(raw) || raw.length === 0) throw new Error("top_features must be a non-empty array");
  const features = raw.map((feature) => {
    const valid =
      feature &&
      typeof feature.name === "string" &&
      typeof feature.label === "string" &&
      (feature.direction === "up" || feature.direction === "down") &&
      Number.isFinite(feature.size) &&
      feature.size >= 0;
    if (!valid) throw new Error("Invalid feature: " + JSON.stringify(feature));
    return feature;
  });
  return features.sort((a, b) => b.size - a.size).slice(0, MAX_FEATURES);
}

function parsePrediction(raw) {
  if (!raw || !Number.isFinite(raw.probability) || raw.probability < 0 || raw.probability > 1) {
    throw new Error("probability must be a number from 0 to 1");
  }
  return { probability: raw.probability, top_features: parseFeatures(raw.top_features) };
}

function formatPercent(probability) {
  const percent = Math.round(probability * 100);
  if (percent < 1) return "<1%";
  if (percent > 99) return ">99%";
  return percent + "%";
}

const isAboveCutoff = (probability) => probability >= SCREEN_CUTOFF;
const capitalize = (text) => text.charAt(0).toUpperCase() + text.slice(1);
const exampleLabel = (example) => example.label;

// Colors: what the model found follows the score, and a speaker's true group follows the dataset's label.
const findingTone = (probability) => (isAboveCutoff(probability) ? "tone-paralysis" : "tone-healthy");
const groupTone = (example) => (example.group === "paralysis" ? "tone-paralysis" : "tone-healthy");

/* ---------- Result panel ---------- */

function scoreScale(probability) {
  const cutoffPercent = SCREEN_CUTOFF * 100;
  return el("div", { class: "score-scale", "aria-hidden": "true" },
    el("div", { class: "score-track" }, el("div", { class: "score-fill", style: `width:${probability * 100}%` })),
    el("div", { class: "score-cutoff", style: `left:${cutoffPercent}%` }),
    el("div", { class: "score-end score-end-low" }, "0%"),
    el("div", { class: "score-cutoff-label", style: `left:${cutoffPercent}%` }, `Cutoff ${Math.round(cutoffPercent)}%`),
    el("div", { class: "score-end score-end-high" }, "100%"));
}

function scoreBlock(probability, isSample = false) {
  const finding = isAboveCutoff(probability) ? "Vocal fold paralysis pattern detected" : "No vocal fold paralysis pattern detected";
  return el("div", { class: `score ${findingTone(probability)}` },
    el("div", { class: "score-row" },
      el("div", { class: "score-number" }, formatPercent(probability)),
      el("div", { class: "score-text" },
        el("div", { class: "score-finding" }, isSample ? `Sample output: ${finding.charAt(0).toLowerCase()}${finding.slice(1)}` : finding))),
    scoreScale(probability),
    el("p", { class: "score-explain" },
      "How closely this voice resembles the paralysis group in the training data."));
}

function scoreTile(title, source, probability) {
  return el("div", { class: `score-tile ${findingTone(probability)}` },
    el("div", { class: "score-tile-title" }, title),
    el("div", { class: "score-tile-source" }, source),
    el("div", { class: "score-number" }, formatPercent(probability)),
    el("div", { class: "score-finding" }, isAboveCutoff(probability) ? "Paralysis pattern detected" : "No paralysis pattern detected"),
    scoreScale(probability));
}

function featuresBlock(features, title = "Top features driving this score") {
  const largest = Math.max(...features.map((feature) => feature.size)) || 1;
  const rows = features.map((feature) => {
    const up = feature.direction === "up";
    const width = (feature.size / largest) * 100;
    const effect = up ? "raises the score" : "lowers the score";
    return el("li", { class: "feature", title: `${feature.label} ${effect}. Effect size: ${feature.size.toFixed(2)}` },
      el("span", { class: "feature-label" }, feature.label,
        el("span", { class: "visually-hidden" }, `: ${effect}`)),
      el("span", { class: "feature-bar", "aria-hidden": "true" },
        el("span", { class: "bar " + (up ? "bar-up" : "bar-down"), style: `width:calc(${width}% - 2px)` })));
  });
  return el("div", { class: "features" },
    el("div", { class: "features-head" },
      el("h3", { style: "font-size:18px" }, title),
      el("div", { class: "features-poles", "aria-hidden": "true" },
        el("span", { class: "pole" }, el("span", { class: "pole-swatch", style: "background:var(--lowers)" }), "◀ Lowers score"),
        el("span", { class: "pole" }, "Raises score ▶", el("span", { class: "pole-swatch", style: "background:var(--raises)" })))),
    el("ul", { class: "feature-list" }, ...rows),
    el("p", { class: "features-note" }, "Longer bars had more effect on this score. They show what the model used, not what caused a voice problem."));
}

function renderExampleResult(example) {
  const { result } = example;
  resultEl.replaceChildren(
    el("div", { class: "result-head" },
      el("h2", { class: `tone-text ${groupTone(example)}` }, exampleLabel(example))),
    el("audio", { class: "example-audio", controls: "", src: `examples/${encodeURIComponent(example.file)}`, "aria-label": "Play this recording" }),
    scoreTile("Live score", "Deployed model, trained on these speakers", result.probability),
    featuresBlock(result.top_features, "Top features: live score"));
}

function renderRecordingResult(prediction) {
  const mockNote = USE_MOCK
    ? [el("p", { class: "mock-note" }, "This is a fixed sample response. Your recording was not analyzed.")]
    : [];
  resultEl.replaceChildren(
    el("div", { class: "result-head" },
      el("h2", {}, "Your recording"),
      USE_MOCK ? el("span", { class: "badge" }, "MOCK RESULT") : null),
    el("p", { class: "result-meta" }, `${capitalize(prediction.sex)}. ${prediction.audioName}`),
    ...mockNote,
    scoreBlock(prediction.data.probability, USE_MOCK),
    featuresBlock(prediction.data.top_features));
}

function renderMessage(title, text) {
  resultEl.replaceChildren(el("div", { class: "state" }, el("h2", {}, title), el("p", { class: "muted" }, text)));
}

function renderLoading(slow) {
  resultEl.replaceChildren(el("div", { class: "state", role: "status" },
    el("div", { class: "spinner", "aria-hidden": "true" }),
    el("h2", {}, slow ? "Still working…" : "Analyzing the recording…"),
    el("p", { class: "muted" }, slow
      ? "The scoring service may be starting up. This can take up to 30 seconds."
      : "This usually takes a few seconds.")));
}

function renderPredictionError(message) {
  resultEl.replaceChildren(el("div", { class: "state state-error", role: "alert" },
    el("h2", {}, "We couldn’t get a score"),
    el("p", {}, message),
    el("div", { class: "state-actions" },
      el("button", { class: "button", type: "button", onclick: submitAudio }, "Try again"),
      state.mode === "recording" && state.examples
        ? el("button", { class: "button", type: "button", onclick: () => setMode("examples") }, "View example speakers")
        : null)));
}

function renderResult() {
  if (state.mode === "examples") {
    const example = state.selectedExample;
    if (example) renderExampleResult(example);
    else if (state.examplesError) renderMessage("Examples unavailable", "You can still try a recording.");
    else renderMessage("Loading…", "Getting the example speakers.");
    return;
  }
  const { status } = state.prediction;
  if (status === "loading") renderLoading(state.prediction.slow);
  else if (status === "error") renderPredictionError(state.prediction.message);
  else if (status === "done") renderRecordingResult(state.prediction);
  else renderMessage("Your score will appear here", "Choose sex, then drag and drop a WAV file or record your voice.");
}

/* ---------- Example speakers ---------- */

function parseExamples(raw) {
  if (!raw || !Array.isArray(raw.examples) || !raw.examples.length) throw new Error("examples must be a non-empty array");
  const baseLabel = (example) =>
    `${capitalize(example.sex)}, ${example.group === "paralysis" ? "vocal fold paralysis" : "healthy"}`;
  const totals = {};
  const seen = {};
  for (const example of raw.examples) totals[baseLabel(example)] = (totals[baseLabel(example)] || 0) + 1;
  return raw.examples.map((example) => {
    const valid = typeof example.file === "string" &&
      ["female", "male"].includes(example.sex) &&
      ["healthy", "paralysis"].includes(example.group);
    if (!valid) throw new Error("Invalid example: " + example.file);
    // Speakers with the same sex and group are numbered, so their labels stay distinct.
    const base = baseLabel(example);
    seen[base] = (seen[base] || 0) + 1;
    const label = totals[base] > 1 ? `${base} ${seen[base]}` : base;
    return { ...example, label, result: parsePrediction(example.result) };
  });
}

async function loadExamples() {
  try {
    const response = await fetch(EXAMPLES_URL);
    if (!response.ok) throw new Error("HTTP " + response.status);
    state.examples = parseExamples(await response.json());
    state.selectedExample = state.examples[0];
  } catch (error) {
    console.error("Could not load examples:", error);
    state.examplesError = location.protocol === "file:"
      ? "This page must run from a local web server. From the repo folder, run: python dev_server.py 8001. Then open http://localhost:8001."
      : "The example speakers could not be loaded. Please reload the page.";
  }
  renderExamples();
  renderResult();
}

function exampleStatus(example) {
  const { probability } = example.result;
  return `${isAboveCutoff(probability) ? "Paralysis pattern" : "No paralysis pattern"} · ${formatPercent(probability)}`;
}

function renderExamples() {
  if (state.examplesError) {
    $("examples-list").replaceChildren(el("div", { class: "inline-error", role: "alert" }, state.examplesError));
    return;
  }
  const groups = ["female", "male"].map((sex) => {
    const rows = (state.examples || []).filter((example) => example.sex === sex);
    if (!rows.length) return null;
    return el("div", { class: "example-group" },
      el("h3", {}, capitalize(sex)),
      ...rows.map((example) =>
        el("button", {
          class: `example ${findingTone(example.result.probability)}`,
          type: "button",
          "aria-pressed": String(example === state.selectedExample),
          onclick: () => selectExample(example),
        },
        el("span", { class: `tone-text ${groupTone(example)}` }, exampleLabel(example)),
        el("span", { class: "example-status tone-text" }, exampleStatus(example)))));
  });
  $("examples-list").replaceChildren(...groups.filter(Boolean));
}

function selectExample(example) {
  state.selectedExample = example;
  renderExamples();
  renderResult();
  // In the one-column layout the result sits below the list.
  if (window.matchMedia("(max-width: 900px)").matches) {
    resultEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

/* ---------- Tabs ---------- */

function setMode(mode, moveFocus = false) {
  if (mode !== "recording") cancelRecording();
  state.mode = mode;
  for (const name of ["examples", "recording"]) {
    const tab = $("tab-" + name);
    const selected = name === mode;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    $("panel-" + name).hidden = !selected;
    if (selected && moveFocus) tab.focus();
  }
  renderResult();
}

function handleTabKeys(event) {
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault();
    setMode(state.mode === "examples" ? "recording" : "examples", true);
  }
}

/* ---------- Audio input ---------- */

const selectedSex = () => {
  const checked = document.querySelector('input[name="sex"]:checked');
  return checked ? checked.value : null;
};

function showAudioError(message) {
  $("audio-error").textContent = message || "";
  $("audio-error").hidden = !message;
}

function updateSubmitState() {
  const sex = selectedSex();
  const busy = state.prediction.status === "loading" || activeRecording !== null;
  $("submit-button").disabled = !sex || !state.audio || busy;
  let hint = "";
  if (!sex && !state.audio) hint = "Choose sex and add a recording to continue.";
  else if (!sex) hint = "Choose sex to get a result.";
  else if (!state.audio) hint = "Add a recording to continue.";
  $("submit-hint").textContent = hint;
  $("submit-hint").hidden = !hint;
}

/** Any change to the inputs makes the current score out of date, so it is cleared. */
function resetPrediction() {
  if (activeRequest) activeRequest.abort();
  state.prediction = { status: "idle" };
  renderResult();
  updateSubmitState();
}

function setAudio(blob, name, autoScore = false) {
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = blob ? URL.createObjectURL(blob) : null;
  state.audio = blob ? { blob, name, autoScore } : null;
  $("audio-summary").hidden = !blob;
  if (blob) {
    $("audio-name").textContent = name;
    $("audio-preview").src = audioUrl;
  } else {
    $("audio-preview").removeAttribute("src");
  }
  resetPrediction();
}

async function isWavFile(file) {
  let header;
  try {
    header = new Uint8Array(await file.slice(0, 12).arrayBuffer());
  } catch {
    return false; // For example, a folder dropped on the page.
  }
  const text = (start, end) => String.fromCharCode(...header.slice(start, end));
  return text(0, 4) === "RIFF" && text(8, 12) === "WAVE";
}

/** What is wrong with an audio file, or null if it can be sent. */
async function audioFileProblem(file) {
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    return `That file is larger than ${MAX_UPLOAD_MB} MB. A 2 second recording should be much smaller.`;
  }
  if (!(await isWavFile(file))) {
    return "That doesn’t look like a WAV file. Please use a .wav file, or record in the browser.";
  }
  return null;
}

async function handleFile(file) {
  cancelRecording();
  showAudioError(null);
  const problem = await audioFileProblem(file);
  if (problem) {
    showAudioError(problem);
    return;
  }
  setAudio(file, file.name, true);
  submitAudio(); // Waits for sex if it hasn't been chosen yet.
}

function handleFileChosen(event) {
  const input = event.target;
  const file = input.files[0];
  input.value = ""; // Lets the same file be chosen again.
  if (file) handleFile(file);
}

function handleSexChange() {
  resetPrediction();
  if (state.audio && state.audio.autoScore) submitAudio();
}

/** The whole page accepts a drop, so a near miss can't navigate away from the demo. */
function setUpDragAndDrop() {
  let dragDepth = 0;
  const hasFiles = (event) => event.dataTransfer && Array.from(event.dataTransfer.types).includes("Files");
  const setActive = (active) => $("drop-zone").classList.toggle("drop-zone-active", active);

  window.addEventListener("dragenter", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    if (dragDepth === 0 && state.mode !== "recording") setMode("recording");
    dragDepth += 1;
    setActive(true);
  });
  window.addEventListener("dragover", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  });
  window.addEventListener("dragleave", (event) => {
    if (!hasFiles(event)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) setActive(false);
  });
  window.addEventListener("drop", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepth = 0;
    setActive(false);
    if (state.mode !== "recording") setMode("recording");
    const { files } = event.dataTransfer;
    if (files.length !== 1) showAudioError("Please drop one file at a time.");
    else handleFile(files[0]);
  });
}

/* ---------- In-browser recording ---------- */

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeText = (offset, text) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeText(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((sample, index) => {
    const clipped = Math.max(-1, Math.min(1, sample));
    view.setInt16(44 + index * 2, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true);
  });
  return new Blob([buffer], { type: "audio/wav" });
}

function micErrorMessage(error) {
  if (!navigator.mediaDevices || !window.AudioWorkletNode) {
    return "Recording isn’t available here. Open the page from localhost or an https address in a current browser, or upload a WAV file.";
  }
  if (error && (error.name === "NotAllowedError" || error.name === "SecurityError")) {
    return "The browser blocked the microphone. Allow microphone access for this page and try again, or upload a WAV file.";
  }
  if (error && error.name === "NotFoundError") {
    return "No microphone was found. Connect one and try again, or upload a WAV file.";
  }
  return "The recording didn’t work. Please try again, or upload a WAV file.";
}

function setRecorderView(text, progress) {
  $("recorder-status").hidden = false;
  $("recorder-text").textContent = text;
  $("recorder-progress").style.width = `${progress * 100}%`;
}

function finishRecordingSession() {
  if (!activeRecording) return;
  const { stream, context, timers } = activeRecording;
  timers.forEach(clearTimeout);
  if (stream) stream.getTracks().forEach((track) => track.stop());
  if (context) context.close().catch(() => {});
  activeRecording = null;
  $("recorder-status").hidden = true;
  $("record-button").disabled = false;
  updateSubmitState();
}

function cancelRecording() {
  finishRecordingSession();
}

async function startRecording() {
  if (activeRecording) return;
  showAudioError(null);
  setAudio(null);
  if (!navigator.mediaDevices || !window.AudioWorkletNode) {
    showAudioError(micErrorMessage());
    return;
  }

  const session = { stream: null, context: null, timers: [] };
  activeRecording = session;
  $("record-button").disabled = true;
  updateSubmitState();
  setRecorderView("Allow the microphone if asked…", 0);

  try {
    // Created during the click, because Safari can otherwise leave it suspended.
    const context = new AudioContext();
    session.context = context;

    // Browser voice processing would change the acoustic measures, so it is turned off.
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    session.stream = stream;
    if (activeRecording !== session) { stream.getTracks().forEach((track) => track.stop()); return; }

    await context.audioWorklet.addModule("recorder-worklet.js");
    if (activeRecording !== session) return;
    await context.resume();

    const source = context.createMediaStreamSource(stream);
    const recorder = new AudioWorkletNode(context, "recorder-processor");
    const mute = context.createGain();
    mute.gain.value = 0; // Keeps the graph running without playing the mic through the speakers.
    // Discards countdown audio. Without a handler it would queue up and fill the recording instantly.
    recorder.port.onmessage = () => {};
    source.connect(recorder).connect(mute).connect(context.destination);

    await runCountdown(session);
    if (activeRecording !== session) return;
    const samples = await captureSamples(session, recorder, context.sampleRate);
    if (activeRecording !== session) return;

    const sampleRate = context.sampleRate;
    finishRecordingSession();
    const peak = samples.reduce((max, sample) => Math.max(max, Math.abs(sample)), 0);
    if (peak < SILENCE_PEAK) {
      showAudioError("We didn’t hear any sound. Check that the right microphone is selected, then try again.");
      return;
    }
    setAudio(encodeWav(samples, sampleRate), `Browser recording (${RECORD_SECONDS} seconds)`);
  } catch (error) {
    console.error("Recording failed:", error);
    if (activeRecording === session) {
      finishRecordingSession();
      showAudioError(micErrorMessage(error));
    }
  }
}

function runCountdown(session) {
  return new Promise((resolve) => {
    let remaining = COUNTDOWN_SECONDS;
    const tick = () => {
      if (activeRecording !== session) { resolve(); return; }
      if (remaining === 0) { resolve(); return; }
      setRecorderView(`Get ready… ${remaining}`, 0);
      remaining -= 1;
      session.timers.push(setTimeout(tick, 1000));
    };
    tick();
  });
}

function captureSamples(session, recorder, sampleRate) {
  return new Promise((resolve, reject) => {
    const target = Math.round(RECORD_SECONDS * sampleRate);
    const samples = new Float32Array(target);
    let filled = 0;
    setRecorderView("Say “ahh” now", 0);
    // If the audio stops arriving, fail instead of waiting forever.
    session.timers.push(setTimeout(() => reject(new Error("Recording timed out")), (RECORD_SECONDS + 3) * 1000));
    recorder.port.onmessage = ({ data }) => {
      if (activeRecording !== session) return;
      const chunk = data.subarray(0, target - filled);
      samples.set(chunk, filled);
      filled += chunk.length;
      $("recorder-progress").style.width = `${(filled / target) * 100}%`;
      if (filled >= target) {
        recorder.port.onmessage = null;
        resolve(samples);
      }
    };
  });
}

/* ---------- Scoring requests ---------- */

const wait = (ms, signal) => new Promise((resolve, reject) => {
  const timer = setTimeout(resolve, ms);
  signal.addEventListener("abort", () => {
    clearTimeout(timer);
    reject(new DOMException("Aborted", "AbortError"));
  }, { once: true });
});

async function requestMock(signal) {
  await wait(MOCK_SCENARIO === "slow" ? SLOW_NOTICE_MS + 3000 : MOCK_DELAY_MS, signal);
  if (MOCK_SCENARIO === "error") throw new UserFacingError("The scoring service reported a problem. (Simulated mock error.)");
  const response = await fetch(MOCK_RESPONSE_URL, { signal });
  return response.json();
}

async function requestApi(audio, sex, signal) {
  const body = new FormData();
  body.append("audio", audio.blob, "recording.wav"); // A fixed name, because filenames can describe the speaker.
  body.append("sex", sex);
  const response = await fetch(API_URL, { method: "POST", body, signal });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = payload && typeof payload.error === "string" ? payload.error : null;
    throw new UserFacingError(message || `The scoring service had a problem (error ${response.status}). Please try again.`);
  }
  return payload;
}

class UserFacingError extends Error {}

async function submitAudio() {
  const sex = selectedSex();
  const audio = state.audio;
  if (!sex || !audio) return;
  if (activeRequest) activeRequest.abort();

  const controller = new AbortController();
  activeRequest = controller;
  let timedOut = false;
  const slowTimer = setTimeout(() => {
    state.prediction = { status: "loading", slow: true };
    if (state.mode === "recording") renderResult();
  }, SLOW_NOTICE_MS);
  const timeoutTimer = setTimeout(() => { timedOut = true; controller.abort(); }, REQUEST_TIMEOUT_MS);

  state.prediction = { status: "loading", slow: false };
  if (state.mode === "recording") renderResult();
  updateSubmitState();

  let next;
  try {
    const raw = USE_MOCK ? await requestMock(controller.signal) : await requestApi(audio, sex, controller.signal);
    next = { status: "done", data: parsePrediction(raw), sex, audioName: audio.name };
  } catch (error) {
    if (error.name === "AbortError" && !timedOut) next = null; // Replaced by a newer action.
    else if (timedOut) next = { status: "error", message: "The scoring service took too long to respond. Please try again." };
    else if (error instanceof UserFacingError) next = { status: "error", message: error.message };
    else {
      console.error("Scoring failed:", error);
      next = { status: "error", message: "We couldn’t reach the scoring service, or its response wasn’t usable. Please try again." };
    }
  } finally {
    clearTimeout(slowTimer);
    clearTimeout(timeoutTimer);
  }

  if (activeRequest !== controller || !next) return;
  activeRequest = null;
  state.prediction = next;
  if (state.mode === "recording") renderResult();
  updateSubmitState();
  if (state.mode === "recording" && window.matchMedia("(max-width: 900px)").matches) {
    resultEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

/* ---------- Start-up ---------- */

function init() {
  $("mock-banner").hidden = !USE_MOCK;
  $("privacy-note").textContent = USE_MOCK
    ? "In mock mode, your audio stays in this browser."
    : "Your audio is sent to the model and isn’t saved.";

  $("tab-examples").addEventListener("click", () => setMode("examples"));
  $("tab-recording").addEventListener("click", () => setMode("recording"));
  $("tab-examples").addEventListener("keydown", handleTabKeys);
  $("tab-recording").addEventListener("keydown", handleTabKeys);

  document.querySelectorAll('input[name="sex"]').forEach((input) => input.addEventListener("change", handleSexChange));
  $("file-input").addEventListener("change", handleFileChosen);
  setUpDragAndDrop();
  $("record-button").addEventListener("click", startRecording);
  $("recorder-cancel").addEventListener("click", cancelRecording);
  $("submit-button").addEventListener("click", submitAudio);

  updateSubmitState();
  renderResult();
  loadExamples();
}

init();

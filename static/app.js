// Alefba Reader (Persian letter reader) - page behaviour.
// Photo in (file, drop, paste or camera) -> optional drag-selection -> POST /predict -> show the answer.

const $ = id => document.getElementById(id);
const LETTERS = JSON.parse($("letters-data").textContent);   // all 32: id, letter, latin, ascii, name
const isPhone = window.matchMedia("(pointer: coarse)").matches;
const narrow = () => window.innerWidth < 960;                 // one-column layout (phones, small windows)


// Noghte, the mascot: one drawing (the <template> in index.html) copied into each place it appears

const NOGHTE = $("noghte-art").content;
document.querySelectorAll(".noghte").forEach(el => el.append(NOGHTE.cloneNode(true)));

function mood(el, m) { el.dataset.mood = m; }


// light / dark colours (a choice made with the button is remembered in this browser)

function showTheme(t) {
  document.documentElement.dataset.theme = t;
  $("themeBtn").setAttribute("aria-label", t === "dark" ? "Switch to light mode" : "Switch to dark mode");
}

showTheme(document.documentElement.dataset.theme === "dark" ? "dark" : "light");
$("themeBtn").onclick = () => {
  const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  // colours fade over to the new theme (only during the switch, so nothing else is slowed down)
  document.documentElement.classList.add("theming");
  clearTimeout(window.themingDone);
  window.themingDone = setTimeout(() => document.documentElement.classList.remove("theming"), 600);
  showTheme(t);
  try { localStorage.setItem("theme", t); } catch (e) { /* private window: the choice just isn't kept */ }
};


let current = null;            // the photo (File) that will be read
let stream = null;             // live camera stream, while the camera is on
let mirrored = false;          // mirror the camera PREVIEW only (never the saved photo)
let facing = "environment";    // which camera to ask for: "environment" (rear) or "user" (front)
let sel = null;                // box dragged on the photo, in the photo's own pixels
let several = false;           // the last answer held several letters (saved with one choice per letter)
let busy = false;              // a photo is being read right now
const history = [];            // this session's answers: { file, answer }, newest last
let shown = -1;                // which of them is on screen (-1: none, e.g. a letter from the chart)
const SINGLE_FOLDER = "random_test_photos", MULTI_FOLDER = "multi_letter_photos";
const HISTORY_MAX = 10;
const READING_MIN_MS = 900;    // "Reading your photo..." stays at least this long
const PHOTO_MAX_H = 420;       // tallest the chosen photo is drawn (px); small photos are enlarged to fit


// showing and hiding parts of the page

const ANSWER_PARTS = ["placeholder", "reading", "refused", "result", "word"];

function showOnly(id) {                    // one of ANSWER_PARTS; "How it reads" and Save are separate
  ANSWER_PARTS.forEach(p => { $(p).hidden = p !== id; });
  $("howread").hidden = true;
  $("savebox").hidden = true;
  $("clearAnswer").hidden = !["result", "word", "refused"].includes(id);
  $("clearAnswer").querySelector("span").textContent = "Clear";
}

// "x Clear" beside the answer: closes a letter opened from the chart (back to what was there before),
// otherwise clears the photo and its answer, like "x New photo"
let beforeChart = -1;
$("clearAnswer").onclick = () => {
  if ($("result").classList.contains("chart") && !$("result").hidden) {
    const h = history[beforeChart];
    if (h && current === h.file) { showAnswer(h.answer); shown = beforeChart; drawSession(); }
    else hideResults();
  } else {
    newPhoto();
  }
};

function hideResults() {
  showOnly("placeholder");
  shown = -1;
  drawSession();
}

// phones: the answer is below the photo panel, so bring it into view (and offer the way back up)
function revealAnswer() {
  if (!narrow()) return;
  $("answers").scrollIntoView({ behavior: "smooth", block: "start" });
}

const EMPTY_HINT = $("hint").cloneNode(true);   // the "Drop a photo" text, put back when the photo is cleared

function showEmpty() {
  $("drop").replaceChildren(EMPTY_HINT.cloneNode(true));
  $("drop").classList.remove("has-photo");
  $("selrow").hidden = true;
  $("straightRow").hidden = true;
  $("go").disabled = true;
}


// choosing a photo

// the photo as chosen, and how it has been turned on the page (quarter turns clockwise, and a small
// straightening angle): every turn is drawn again from the original, so turning never blurs it
let orig = null;               // { file, img, paper, quarter: 0..3, fine: degrees clockwise }

function setFile(f, edited = false) {
  if (!f) return;
  stopCam();
  current = f;
  if (!edited) {
    orig = { file: f, img: null, paper: null, quarter: 0, fine: 0 };
    $("straight").value = 0;
    showStraight();
    closeStraight();
  }
  $("err").textContent = "";
  hideResults();

  const img = new Image();
  img.src = URL.createObjectURL(f);
  img.alt = "Selected image. Drag on it to select one letter.";
  img.draggable = false;

  const stage = document.createElement("div");
  stage.id = "stage";
  const box = document.createElement("div");
  box.id = "selbox";
  box.hidden = true;
  const clear = document.createElement("button");          // "x New photo": back to the start
  clear.type = "button";
  clear.className = "clear-photo";
  clear.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>New photo';
  clear.onclick = e => { e.stopPropagation(); newPhoto(); };
  const tools = document.createElement("div");             // turn left, turn right, straighten
  tools.className = "turns";
  for (const [label, path, act] of [
      ["Turn left", "M4 12a8 8 0 1 0 2.3-5.7M4 4v5h5", () => turnPhoto(-1)],
      ["Turn right", "M20 12a8 8 0 1 1-2.3-5.7M20 4v5h-5", () => turnPhoto(1)],
      ["Straighten", "M3 19h18M5 19L18 6M14 19a9 9 0 0 0-3-6.7", openStraight]]) {
    const t = document.createElement("button");
    t.type = "button";
    t.className = "turn-photo";
    t.title = label === "Straighten" ? "Straighten a small tilt" : label + " (a quarter turn)";
    t.setAttribute("aria-label", label);
    t.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="' + path + '"/></svg>';
    t.onclick = e => { e.stopPropagation(); act(); };
    tools.append(t);
  }
  stage.append(img, box, clear, tools);

  $("drop").replaceChildren(stage);
  $("drop").classList.add("has-photo");
  img.onload = () => fitPhoto(img);
  setupSelect(img, box);
  clearSel();
  $("selrow").hidden = false;
  $("go").disabled = false;
}

// Turn left / Turn right: a quarter turn (dir -1 or 1); if the photo had been read, it is read again
// (the app straightens tilted ruled paper itself, but a sideways or upside-down letter on plain paper
// can need a hand)
function turnPhoto(dir) {
  if (busy || !orig) return;
  orig.quarter = (orig.quarter + dir + 4) % 4;
  redraw();
}

// Straighten: a slider that takes the place of the tip under the photo while it is open. While it
// moves only the picture on screen turns; when it is let go the photo is drawn again at that angle
// (and read again if it had been read)
function openStraight() {
  $("selrow").hidden = true;
  $("straightRow").hidden = false;
  $("straight").focus();
}

function closeStraight() {
  $("straightRow").hidden = true;
  $("selrow").hidden = !current;
}

function showStraight() {
  const v = Number($("straight").value);
  $("straightVal").textContent = (v > 0 ? "+" : "") + v + "°" + (Math.abs(v) === 45 ? " (most)" : "");
  $("straightReset").hidden = v === 0;
}

// while the slider is held, the cursor stays a grabbing hand wherever the pointer goes (otherwise it
// shows the photo's crosshair or a button's hand, which looks as if the slider stopped working)
$("straight").addEventListener("pointerdown", () => document.documentElement.classList.add("sliding"));
for (const ev of ["pointerup", "pointercancel"])
  window.addEventListener(ev, () => document.documentElement.classList.remove("sliding"));

$("straight").oninput = () => {
  showStraight();
  const img = $("drop").querySelector("#stage img");
  if (img && orig) img.style.transform = `rotate(${Number($("straight").value) - orig.fine}deg)`;
};
$("straight").onchange = () => {
  if (!orig || busy || Number($("straight").value) === orig.fine) return;
  orig.fine = Number($("straight").value);
  redraw();
};
$("straightReset").onclick = () => {
  $("straight").value = 0;
  showStraight();
  $("straight").onchange();
};
$("straightDone").onclick = closeStraight;

// the original photo drawn at quarter x 90 + fine degrees (clockwise) on a canvas big enough for all of
// it; new corners get the photo's own paper colour (the colour along its edges), as if the paper went on
async function redraw() {
  const wasRead = shown >= 0 && history[shown] && history[shown].file === current;
  const open = !$("straightRow").hidden;
  const angle = orig.quarter * 90 + orig.fine;
  if (angle === 0) {
    setFile(orig.file, true);
  } else {
    if (!orig.img) {
      orig.img = new Image();
      orig.img.src = URL.createObjectURL(orig.file);
      await orig.img.decode();
      orig.paper = paperColour(orig.img);
    }
    const w = orig.img.naturalWidth, h = orig.img.naturalHeight, r = angle * Math.PI / 180;
    const c = document.createElement("canvas");
    c.width = Math.round(Math.abs(w * Math.cos(r)) + Math.abs(h * Math.sin(r)));
    c.height = Math.round(Math.abs(w * Math.sin(r)) + Math.abs(h * Math.cos(r)));
    const g = c.getContext("2d");
    g.fillStyle = orig.paper;
    g.fillRect(0, 0, c.width, c.height);
    g.translate(c.width / 2, c.height / 2);
    g.rotate(r);
    g.drawImage(orig.img, -w / 2, -h / 2);
    const blob = await new Promise(ok => c.toBlob(ok, "image/jpeg", 0.95));
    const name = (orig.file.name ? orig.file.name.replace(/\.[^.]*$/, "") : "photo") + "_turned.jpg";
    setFile(new File([blob], name, { type: "image/jpeg" }), true);
  }
  if (open) openStraight();
  if (wasRead) $("go").click();
}

// the colour along the photo's edges (the middle value of each channel): its paper
function paperColour(img) {
  const c = document.createElement("canvas");
  c.width = 64; c.height = 64;
  const g = c.getContext("2d");
  g.drawImage(img, 0, 0, 64, 64);
  const d = g.getImageData(0, 0, 64, 64).data, ch = [[], [], []];
  for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) {
    if (x > 1 && x < 62 && y > 1 && y < 62) continue;
    const k = (y * 64 + x) * 4;
    for (let i = 0; i < 3; i++) ch[i].push(d[k + i]);
  }
  const mid = a => a.sort((p, q) => p - q)[a.length >> 1];
  return `rgb(${mid(ch[0])}, ${mid(ch[1])}, ${mid(ch[2])})`;
}

// Draw the photo as big as the panel allows (small photos and screenshots are enlarged), keeping its
// shape, so there is room to drag a box on it. The box maths uses the drawn size, so it stays exact.
function fitPhoto(img) {
  if (!img || !img.naturalWidth) return;
  const avail = $("drop").clientWidth;
  const s = Math.min(avail / img.naturalWidth, PHOTO_MAX_H / img.naturalHeight);
  img.style.width = Math.round(img.naturalWidth * s) + "px";
  img.style.height = Math.round(img.naturalHeight * s) + "px";
}

window.addEventListener("resize", () => fitPhoto($("drop").querySelector("img")));

function newPhoto() {
  stopCam();
  current = null;
  sel = null;
  $("err").textContent = "";
  $("file").value = "";
  showEmpty();
  hideResults();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

$("file").onchange = e => setFile(e.target.files[0]);
$("cam").onchange = e => setFile(e.target.files[0]);

$("drop").onclick = () => { if (!stream && !current) $("file").click(); };
$("drop").onkeydown = e => {
  if (e.target === $("drop") && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); $("file").click(); }
};

// drag a photo anywhere over the page: the whole page becomes the drop area
let dragDepth = 0;
const hasFiles = e => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
document.addEventListener("dragenter", e => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth++;
  $("dropAll").hidden = false;
});
document.addEventListener("dragover", e => { if (hasFiles(e)) e.preventDefault(); });
document.addEventListener("dragleave", e => {
  if (!hasFiles(e)) return;
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) $("dropAll").hidden = true;
});
document.addEventListener("drop", e => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  $("dropAll").hidden = true;
  const f = [...e.dataTransfer.files].find(x => x.type.startsWith("image/"));
  if (f) setFile(f);
  else $("err").textContent = "That is not an image. Drop a photo (JPG, PNG...).";
});

// Ctrl+V anywhere on the page: a copied image or a screenshot becomes the photo
// (text pasted into the page is ignored, and so is pasting while a list is being used)
document.addEventListener("paste", e => {
  if (e.target.closest && e.target.closest("select, input, textarea")) return;
  const item = [...(e.clipboardData ? e.clipboardData.items : [])].find(i => i.type.startsWith("image/"));
  if (!item) return;
  const blob = item.getAsFile();
  if (!blob) return;
  e.preventDefault();
  const ext = (blob.type.split("/")[1] || "png").replace("jpeg", "jpg");
  setFile(new File([blob], "pasted." + ext, { type: blob.type }));
});

// Enter reads the photo (unless a button, list or link has the keyboard focus: Enter belongs to it)
document.addEventListener("keydown", e => {
  if (e.key !== "Enter" || e.repeat || $("chart").open) return;
  if (e.target.closest && e.target.closest("button, select, input, textarea, a, summary, #drop")) return;
  if (current && !$("go").disabled) { e.preventDefault(); $("go").click(); }
});


// drag a box on the photo to choose one letter

function clearSel() {
  sel = null;
  const box = $("selbox");
  if (box) box.hidden = true;
  $("clearSel").hidden = true;
  $("selhint").textContent = "Tip: drag a box around one letter on the photo to read only that letter.";
}

function setupSelect(img, box) {
  let start = null;

  const pos = e => {
    const r = img.getBoundingClientRect();
    return [Math.min(Math.max(e.clientX - r.left, 0), r.width),
            Math.min(Math.max(e.clientY - r.top, 0), r.height)];
  };
  const draw = (a, b) => {
    Object.assign(box.style, {
      left: Math.min(a[0], b[0]) + "px", top: Math.min(a[1], b[1]) + "px",
      width: Math.abs(a[0] - b[0]) + "px", height: Math.abs(a[1] - b[1]) + "px",
    });
    box.hidden = false;
  };

  img.addEventListener("pointerdown", e => {
    e.preventDefault();
    e.stopPropagation();
    img.setPointerCapture(e.pointerId);
    start = pos(e);
    draw(start, start);
  });
  img.addEventListener("pointermove", e => { if (start) draw(start, pos(e)); });
  img.addEventListener("pointerup", e => {
    if (!start) return;
    const a = start, b = pos(e);
    start = null;
    const w = Math.abs(a[0] - b[0]), h = Math.abs(a[1] - b[1]);
    if (w < 8 || h < 8) { clearSel(); return; }            // a click, not a drag
    const k = img.naturalWidth / img.clientWidth;           // screen pixels -> photo pixels
    sel = { x: Math.round(Math.min(a[0], b[0]) * k), y: Math.round(Math.min(a[1], b[1]) * k),
            w: Math.round(w * k), h: Math.round(h * k) };
    $("clearSel").hidden = false;
    $("selhint").textContent = "Only the box will be read.";
    hideResults();
  });
  img.addEventListener("click", e => e.stopPropagation());
}

$("clearSel").onclick = () => { clearSel(); hideResults(); };
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && sel && !$("chart").open) { clearSel(); hideResults(); }
});


// camera

function camLabel(text) { $("camBtn").querySelector("span").textContent = text; }   // keeps the icon

function showCamControls(on) {
  $("camBtn").classList.toggle("live", on);
  $("camOff").hidden = !on;
  $("mirrorBtn").hidden = !on;
  $("flipBtn").hidden = !on;
}

function applyMirror() {
  const v = $("drop").querySelector("video");
  if (v) v.style.transform = mirrored ? "scaleX(-1)" : "none";
  $("mirrorBtn").textContent = "Mirror preview: " + (mirrored ? "on" : "off");
  $("mirrorBtn").setAttribute("aria-pressed", mirrored);
}

function stopCam() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  camLabel("Take photo");
  showCamControls(false);
}

async function openStream() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  let lastErr;
  for (const c of [{ video: { facingMode: { ideal: facing } } }, { video: true }]) {
    try { stream = await navigator.mediaDevices.getUserMedia(c); break; } catch (e) { lastErr = e; }
  }
  if (!stream) throw lastErr;

  const v = document.createElement("video");
  v.autoplay = true;
  v.muted = true;
  v.playsInline = true;
  v.setAttribute("playsinline", "");
  v.srcObject = stream;
  $("drop").replaceChildren(v);
  $("drop").classList.add("has-photo");
  $("selrow").hidden = true;
  $("straightRow").hidden = true;

  // follow what the browser actually gave us (laptops have no rear camera)
  const fm = stream.getVideoTracks()[0].getSettings().facingMode;
  if (fm) facing = fm;
  // preview un-mirrored by default, so a letter held up to the camera reads the right way round
  mirrored = false;
  applyMirror();
  try { await v.play(); } catch (e) { /* autoplay may be blocked; the next frame starts it */ }
}

function captureFrame() {
  const v = $("drop").querySelector("video");
  if (!v || !v.videoWidth) {
    $("err").textContent = "Camera is still starting, try again in a moment.";
    return;
  }
  const c = document.createElement("canvas");
  c.width = v.videoWidth;
  c.height = v.videoHeight;
  // never mirror the saved photo: a mirrored Persian letter looks like a different letter
  c.getContext("2d").drawImage(v, 0, 0);
  c.toBlob(b => setFile(new File([b], "capture.jpg", { type: "image/jpeg" })), "image/jpeg", 0.95);
}

$("camBtn").onclick = async () => {
  if (stream) { captureFrame(); return; }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    // browsers only allow a live camera on http://127.0.0.1 / localhost (or https)
    if (isPhone) {
      $("cam").click();               // the phone's camera app takes the photo and it comes straight back
    } else {
      $("err").textContent = "The live camera only works when this page is opened at http://127.0.0.1:"
        + (location.port || "5000") + " on this laptop (browsers block cameras on other addresses). "
        + "Open that address, or use Choose file.";
    }
    return;
  }
  try {
    await openStream();
    current = null;
    camLabel("Capture");
    showCamControls(true);
    $("go").disabled = true;
    $("err").textContent = "";
    hideResults();
  } catch (e) {
    stream = null;
    showCamControls(false);
    const why = e && e.name === "NotAllowedError" ? "Camera permission was denied (allow it in the address bar)."
              : e && e.name === "NotReadableError" ? "The camera is being used by another app. Close it and try again."
              : "No camera was found.";
    $("err").textContent = why + " You can also use Choose file.";
    if (isPhone) $("cam").click();
  }
};

$("camOff").onclick = () => { stopCam(); showEmpty(); };

$("mirrorBtn").onclick = () => { mirrored = !mirrored; applyMirror(); };

$("flipBtn").onclick = async () => {
  facing = facing === "user" ? "environment" : "user";
  try { await openStream(); } catch (e) { $("err").textContent = "Could not switch camera."; }
};


// talking to the app (python app.py)

// POST a form and return the JSON answer. Turns the browser's bare "Failed to fetch" (the app is not
// running, the laptop slept, the Wi-Fi dropped) and non-JSON error pages into messages a person can act on.
async function send(url, fd) {
  let r;
  try {
    r = await fetch(url, { method: "POST", body: fd });
  } catch (e) {
    if (document.body.hasAttribute("data-public")) {       // the online copy: no laptop or terminal involved
      throw new Error("Could not reach the app. Check your internet connection, then press the button again. "
        + "(If the site was asleep, it can take about a minute to wake up.)");
    }
    throw new Error("Could not reach the app. Check that 'python app.py' is still running in the "
      + "terminal (start it again if it stopped), then press the button again."
      + (isPhone ? " On a phone, also check that it is on the same Wi-Fi as the laptop." : ""));
  }
  let d;
  try { d = await r.json(); } catch (e) { d = {}; }
  if (!r.ok) {
    throw new Error(d.error || (r.status === 413 ? "This photo is too big. Send a smaller copy."
                                                 : "The app hit a problem reading this photo (error " + r.status + ")."));
  }
  return d;
}


// a letter inside a word: Persian letters change shape with their neighbours (joined with a zero-width
// joiner, U+200D, so the font draws each form). These 7 never join the letter after them: 2 forms only.

const ZWJ = "‍";
const NON_JOINING = "اآدذرزژو";

function showForms(t) {
  const L = t.letter;
  const forms = NON_JOINING.includes(L)
    ? [[L, "alone or at the start"], [ZWJ + L, "in the middle or at the end"]]
    : [[L, "alone"], [L + ZWJ, "at the start"], [ZWJ + L + ZWJ, "in the middle"], [ZWJ + L, "at the end"]];
  $("formsTitle").textContent = "Inside a word, " + t.name + " looks like this";
  $("formsList").replaceChildren(...forms.map(([shape, where]) => {
    const cell = document.createElement("div");
    const s = document.createElement("span");
    s.lang = "fa";
    s.textContent = shape;
    const w = document.createElement("small");
    w.textContent = where;
    cell.append(s, w);
    return cell;
  }));
  $("formsList").classList.toggle("two", forms.length === 2);
  $("formsNote").textContent = forms.length === 2
    ? t.name.charAt(0).toUpperCase() + t.name.slice(1) + " never joins the letter after it, so it has only two shapes."
    : "";
}


// reading the letter

function setLetter(t) {                     // the letter, its spelling and name (answers and the chart)
  $("letter").textContent = t.letter;
  const span = document.createElement("span");
  span.textContent = t.latin;
  $("latin").replaceChildren(span);
  $("latin").classList.toggle("high", t.latin === "ʿ");      // eyn = ʿ, a small raised mark
  $("name").textContent = t.name;
  $("plain").textContent = t.ascii !== t.latin ? "plain keyboard: " + t.ascii : "";
  $("copyBtn").querySelector("span").textContent = "Copy";
  $("copyBtn").dataset.text = t.latin;
  showForms(t);
}

function showAnswer(d) {
  if (d.letters && d.letters.length > 1) { showWord(d); return; }
  $("views").replaceChildren();
  $("view").hidden = false;
  const [t, ...rest] = d.results;
  setLetter(t);

  const pct = Math.round(t.confidence * 100);
  const low = t.confidence < 0.6;
  const unsure = t.confidence < 0.5;               // shown as a best guess, never as a sure answer
  $("result").classList.remove("chart");
  $("conf").textContent = pct + "% sure" + (low ? " (low: retake the photo)" : "");
  $("unsure").hidden = !unsure;
  $("talk").hidden = unsure;
  $("bestGuess").hidden = !unsure;
  $("result").classList.toggle("unsure", unsure);
  $("result").classList.toggle("low", low);
  $("confbar").firstElementChild.style.width = pct + "%";
  $("confbar").classList.toggle("low", low);
  mood($("resultMascot"), low ? "thinking" : "happy");
  $("bubble").textContent = low ? "Probably " + t.name + ", but a clearer photo would help."
                                : "That's " + t.name + "! I'm " + pct + "% sure.";

  const also = rest.filter(x => x.confidence >= 0.01);    // 0% guesses are noise
  $("altsBox").hidden = !also.length;                     // folded until opened (the answer is the top letter)
  $("altsBox").open = false;
  $("alts").replaceChildren(...also.flatMap((x, n) => {
    const fa = document.createElement("b");
    fa.lang = "fa";
    fa.textContent = x.letter;
    return [...(n ? ["  ·  "] : []), fa, " " + x.name + " · " + Math.round(x.confidence * 100) + "%"];
  }));
  const label = document.createElement("b");
  label.textContent = "Read:";
  $("how").replaceChildren(label, " " + d.how);

  $("found").src = d.found;
  $("view").src = d.view;
  $("truth").value = t.id;
  setSaveMode([]);
  $("saved").textContent = "";
  $("saveBtn").disabled = false;

  showOnly("result");
  $("answerLabel").textContent = "Answer";
  $("howread").hidden = false;
  $("savebox").hidden = false;
}

// a photo with several letters: one tile per letter, in Persian reading order (first letter on the right)
function showWord(d) {
  const tops = d.letters.map(l => l.results[0]);
  $("wordLatin").textContent = tops.map(t => t.latin).join(" ");
  $("tiles").replaceChildren(...tops.map((t, n) => {
    const tile = document.createElement("div");
    tile.className = "mini" + (t.confidence < 0.6 ? " low" : "");
    const num = document.createElement("small");
    num.textContent = n + 1;
    const ch = document.createElement("span");
    ch.className = "mini-letter";
    ch.lang = "fa";
    ch.textContent = t.letter;
    const lat = document.createElement("b");
    lat.textContent = t.latin;
    const conf = document.createElement("small");
    conf.textContent = t.name + " · " + Math.round(t.confidence * 100) + "%";
    tile.append(num, ch, lat, conf);
    return tile;
  }));
  const label = document.createElement("b");
  label.textContent = "Read:";
  $("wordHow").replaceChildren(label, " " + d.how + ". The numbers match the boxes on your photo below.");

  $("found").src = d.found;
  $("view").hidden = true;
  $("views").replaceChildren(...d.letters.map((l, n) => {
    const im = document.createElement("img");
    im.src = l.view;
    im.alt = "Cleaned-up letter " + (n + 1);
    return im;
  }));

  setSaveMode(tops);
  $("saved").textContent = "";
  $("saveBtn").disabled = false;

  showOnly("word");
  $("answerLabel").textContent = "Answer";
  $("howread").hidden = false;
  $("savebox").hidden = false;
}

// Save card: one "really is" choice for a single letter, or one per letter (reading order, first on
// the right) for a photo of several letters, which go to their own folder by default.
function setSaveMode(tops) {
  several = tops.length > 1;
  $("truth").hidden = several;
  $("truthMulti").hidden = !several;
  $("truthLabel").textContent = several ? "These letters really are (right to left):" : "This letter really is:";
  $("truthLabel").htmlFor = several ? "" : "truth";
  $("truthMulti").replaceChildren(...tops.map((t, n) => {
    const s = $("truth").cloneNode(true);
    s.removeAttribute("id");
    s.hidden = false;                  // (the single-letter list it is copied from is hidden right now)
    s.value = t.id;
    s.setAttribute("aria-label", "Letter " + (n + 1));
    return s;
  }));
  const f = $("folder").value;
  if (several && f === SINGLE_FOLDER) $("folder").value = MULTI_FOLDER;
  if (!several && f === MULTI_FOLDER) $("folder").value = SINGLE_FOLDER;
}

// The app refused the photo (messages from predict.py): show it on the answer stage with Noghte.
// Returns false for other problems (camera, file, app not running), which stay under the buttons.
function showRefused(msg) {
  const nothing = msg.match(/^(No (?:complete )?Persian letter found)(?:: | in this photo\. )?(.*)$/);
  const many = msg.startsWith("This photo shows many letters");
  if (!nothing && !many) return false;
  $("refusedTitle").textContent = many ? "Which letter should I read?" : nothing[1];
  const text = many ? msg : nothing[2];
  $("refusedText").textContent = text.charAt(0).toUpperCase() + text.slice(1);
  mood($("refused").querySelector(".noghte"), many ? "thinking" : "puzzled");
  showOnly("refused");
  $("answerLabel").textContent = "Answer";
  return true;
}

$("go").onclick = async () => {
  if (busy || !current) return;
  busy = true;
  $("go").disabled = true;
  $("go").textContent = "Reading...";
  $("err").textContent = "";
  showOnly("reading");
  revealAnswer();
  const file = current;
  const started = performance.now();
  // most photos are read in a fraction of a second: keep "Reading your photo..." up long enough to be seen
  const atLeast = () => new Promise(r => setTimeout(r, Math.max(0, READING_MIN_MS - (performance.now() - started))));
  try {
    const fd = new FormData();
    fd.append("image", file);
    if (sel) {
      fd.append("sel_x", sel.x); fd.append("sel_y", sel.y);
      fd.append("sel_w", sel.w); fd.append("sel_h", sel.h);
    }
    let d;
    try { d = await send("/predict", fd); } finally { await atLeast(); }
    showAnswer(d);
    remember(file, d);
    revealAnswer();                                  // again: the answer is taller than "Reading..."
  } catch (e) {
    if (showRefused(e.message)) {
      revealAnswer();
    } else {
      showOnly("placeholder");
      $("err").textContent = e.message;              // shown under the buttons, so stay up there
      if (narrow()) $("err").scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }
  busy = false;
  $("go").textContent = "Read letter";
  $("go").disabled = !current;
};


// this session: the answers read since the page was opened (kept until the page is reloaded)

function remember(file, answer) {
  history.push({ file, answer });
  if (history.length > HISTORY_MAX) history.shift();
  shown = history.length - 1;
  drawSession();
}

// x beside the session strip: forget this session's answers (the answer on screen stays)
$("clearSession").onclick = () => {
  history.length = 0;
  shown = -1;
  drawSession();
};

function drawSession() {
  $("session").hidden = history.length < 2;          // a strip of one would only repeat the answer
  $("sessionList").replaceChildren(...history.map((h, n) => {
    const tops = (h.answer.letters && h.answer.letters.length > 1)
      ? h.answer.letters.map(l => l.results[0]) : [h.answer.results[0]];
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    if (n === shown) b.setAttribute("aria-current", "true");
    const num = document.createElement("small");
    num.textContent = n + 1;
    const fa = document.createElement("span");
    fa.lang = "fa";
    fa.textContent = tops.map(t => t.letter).join(" ");
    b.append(num, fa, " " + tops.map(t => t.latin).join(" "));
    b.setAttribute("aria-label", "Answer " + (n + 1) + ": " + tops.map(t => t.name).join(", "));
    b.onclick = () => {
      setFile(h.file);             // puts the photo back (and clears the answer) ...
      showAnswer(h.answer);        // ... then shows that answer again
      shown = n;
      drawSession();
      revealAnswer();
    };
    return b;
  }));
}


// copy the spelling (the clipboard needs a secure page; on a phone over Wi-Fi the old way is used)

$("copyBtn").onclick = async () => {
  const text = $("copyBtn").dataset.text || "";
  let ok = false;
  try {
    if (navigator.clipboard && window.isSecureContext) { await navigator.clipboard.writeText(text); ok = true; }
  } catch (e) { ok = false; }
  if (!ok) {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.append(ta);
    ta.select();
    try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
    ta.remove();
  }
  $("copyBtn").querySelector("span").textContent = ok ? "Copied" : "Copy failed";
  $("copyBtn").classList.remove("done");
  void $("copyBtn").offsetWidth;                  // restart the flash on every click
  $("copyBtn").classList.toggle("done", ok);
};


// the alphabet chart: all 32 letters; picking one shows it large, like an answer (no photo involved)

$("chartGrid").replaceChildren(...LETTERS.map(l => {
  const b = document.createElement("button");
  b.type = "button";
  const fa = document.createElement("span");
  fa.lang = "fa";
  fa.textContent = l.letter;
  const lat = document.createElement("b");
  lat.textContent = l.latin;
  const nm = document.createElement("small");
  nm.textContent = l.name;
  b.append(fa, lat, nm);
  b.setAttribute("aria-label", l.name + ", spelled " + l.latin);
  b.onclick = () => { $("chart").close(); showChartLetter(l); };
  return b;
}));

function showChartLetter(l) {
  if (!$("result").classList.contains("chart") || $("result").hidden) beforeChart = shown;
  setLetter(l);
  $("result").classList.add("chart");
  $("result").classList.remove("unsure", "low");
  $("unsure").hidden = true;
  $("bestGuess").hidden = true;
  $("talk").hidden = false;
  mood($("resultMascot"), "idle");
  $("bubble").textContent = "This is " + l.name + ", letter " + (l.id + 1) + " of 32.";
  $("alts").replaceChildren();
  $("altsBox").hidden = true;
  $("how").replaceChildren();
  showOnly("result");
  $("answerLabel").textContent = "From the alphabet chart";
  $("clearAnswer").querySelector("span").textContent = "Close";
  shown = -1;
  drawSession();
  revealAnswer();
}

$("chartBtn").onclick = () => $("chart").showModal();
$("chartClose").onclick = () => $("chart").close();
$("chart").addEventListener("click", e => { if (e.target === $("chart")) $("chart").close(); });   // click outside


// phones: a "New photo" button to jump back up once the photo panel has scrolled out of view

if ("IntersectionObserver" in window) {
  // the top 60 px of the screen don't count: a sliver of the panel left there means it has scrolled away
  new IntersectionObserver(([e]) => { $("toPhoto").hidden = !narrow() || e.isIntersecting; },
                           { rootMargin: "-60px 0px 0px 0px" })
    .observe(document.querySelector(".panel"));
}
$("toPhoto").onclick = () => window.scrollTo({ top: 0, behavior: "smooth" });


// save as test photo

$("saveBtn").onclick = async () => {
  $("saveBtn").disabled = true;
  $("saved").textContent = "Saving...";
  try {
    const fd = new FormData();
    fd.append("image", current);
    if (several) {
      fd.append("class_ids", [...$("truthMulti").querySelectorAll("select")].map(s => s.value).join(","));
    } else {
      fd.append("class_id", $("truth").value);
    }
    fd.append("folder", $("folder").value);
    const d = await send("/save", fd);
    $("saved").textContent = "Saved as " + d.folder + "/" + d.saved
      + ". You can save it again, to another folder or as " + (several ? "other letters." : "another letter.");
  } catch (e) {
    $("saved").textContent = "";
    $("err").textContent = e.message;
  }
  $("saveBtn").disabled = false;      // the same photo can be saved more than once
};

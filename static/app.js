// Alefba Reader (Persian letter reader) - page behaviour.
// Photo in (file, drop, paste or camera) -> optional drag-selection -> POST /predict -> show the answer.

const $ = id => document.getElementById(id);


// Noghte, the mascot: one drawing (the <template> in index.html) copied into each place it appears

const NOGHTE = $("noghte-art").content;
document.querySelectorAll(".noghte").forEach(el => el.append(NOGHTE.cloneNode(true)));

function mood(el, m) { el.dataset.mood = m; }


// light / dark colours (the choice is remembered in this browser)

function setTheme(t) {
  document.documentElement.dataset.theme = t;
  $("themeBtn").setAttribute("aria-label", t === "dark" ? "Switch to light mode" : "Switch to dark mode");
  try { localStorage.setItem("theme", t); } catch (e) { /* private window: the choice just isn't kept */ }
}

setTheme(document.documentElement.dataset.theme === "dark" ? "dark" : "light");
$("themeBtn").onclick = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");

let current = null;            // the photo (File) that will be read
let stream = null;             // live camera stream, while the camera is on
let mirrored = false;          // mirror the camera PREVIEW only (never the saved photo)
let facing = "environment";    // which camera to ask for: "environment" (rear) or "user" (front)
let sel = null;                // box dragged on the photo, in the photo's own pixels
let several = false;           // the last answer held several letters (saved with one choice per letter)
const SINGLE_FOLDER = "random_test_photos", MULTI_FOLDER = "multi_letter_photos";


// showing and hiding parts of the page

function hideResults() {
  $("placeholder").hidden = false;
  $("refused").hidden = true;
  $("result").hidden = true;
  $("word").hidden = true;
  $("howread").hidden = true;
  $("savebox").hidden = true;
}

const EMPTY_HINT = $("hint").cloneNode(true);   // the "Drop a photo" text, put back when the camera turns off

function showEmpty() {
  $("drop").replaceChildren(EMPTY_HINT.cloneNode(true));
  $("selrow").hidden = true;
  $("go").disabled = true;
}


// choosing a photo

function setFile(f) {
  if (!f) return;
  stopCam();
  current = f;
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
  stage.append(img, box);

  $("drop").replaceChildren(stage);
  setupSelect(img, box);
  clearSel();
  $("selrow").hidden = false;
  $("go").disabled = false;
}

$("file").onchange = e => setFile(e.target.files[0]);
$("cam").onchange = e => setFile(e.target.files[0]);

$("drop").onclick = () => { if (!stream && !current) $("file").click(); };
$("drop").onkeydown = e => {
  if (e.target === $("drop") && (e.key === "Enter" || e.key === " ")) $("file").click();
};
$("drop").ondragover = e => { e.preventDefault(); $("drop").classList.add("over"); };
$("drop").ondragleave = () => $("drop").classList.remove("over");
$("drop").ondrop = e => {
  e.preventDefault();
  $("drop").classList.remove("over");
  setFile(e.dataTransfer.files[0]);
};

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
  if (e.key === "Escape" && sel) { clearSel(); hideResults(); }
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
  $("selrow").hidden = true;

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

// phones and tablets: their own camera app can take the photo (the file input with capture=)
const isPhone = window.matchMedia("(pointer: coarse)").matches;

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


// reading the letter

function showAnswer(d) {
  if (d.letters && d.letters.length > 1) { showWord(d); return; }
  $("word").hidden = true;
  $("views").replaceChildren();
  $("view").hidden = false;
  const [t, ...rest] = d.results;

  $("letter").textContent = t.letter;
  const span = document.createElement("span");
  span.textContent = t.latin;
  $("latin").replaceChildren(span);
  $("latin").classList.toggle("high", t.latin === "ʿ");      // eyn = ʿ, a small raised mark
  $("name").textContent = t.name;
  $("plain").textContent = t.ascii !== t.latin ? "plain keyboard: " + t.ascii : "";

  const pct = Math.round(t.confidence * 100);
  const low = t.confidence < 0.6;
  const unsure = t.confidence < 0.5;               // shown as a best guess, never as a sure answer
  $("conf").textContent = pct + "% sure" + (low ? " (low: retake the photo)" : "");
  $("unsure").hidden = !unsure;
  $("result").classList.toggle("unsure", unsure);
  $("result").classList.toggle("low", low);
  $("confbar").firstElementChild.style.width = pct + "%";
  $("confbar").classList.toggle("low", low);

  mood($("resultMascot"), unsure || low ? "thinking" : "happy");
  $("bubble").textContent = unsure ? ""
    : low ? "Probably " + t.name + ", but a clearer photo would help."
    : "That's " + t.name + "! I'm " + pct + "% sure.";

  $("alts").replaceChildren(...rest.filter(x => x.confidence >= 0.01).map(x => {   // 0% guesses are noise
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = x.letter + " " + x.latin + " (" + x.name + ") " + Math.round(x.confidence * 100) + "%";
    return chip;
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

  $("placeholder").hidden = true;
  $("refused").hidden = true;
  $("result").hidden = false;
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

  $("placeholder").hidden = true;
  $("refused").hidden = true;
  $("result").hidden = true;
  $("word").hidden = false;
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
  hideResults();
  $("placeholder").hidden = true;
  $("refusedTitle").textContent = many ? "Which letter should I read?" : nothing[1];
  const text = many ? msg : nothing[2];
  $("refusedText").textContent = text.charAt(0).toUpperCase() + text.slice(1);
  mood($("refused").querySelector(".noghte"), many ? "thinking" : "puzzled");
  $("refused").hidden = false;
  return true;
}

$("go").onclick = async () => {
  $("go").disabled = true;
  $("go").textContent = "Reading...";
  $("err").textContent = "";
  try {
    const fd = new FormData();
    fd.append("image", current);
    if (sel) {
      fd.append("sel_x", sel.x); fd.append("sel_y", sel.y);
      fd.append("sel_w", sel.w); fd.append("sel_h", sel.h);
    }
    showAnswer(await send("/predict", fd));
  } catch (e) {
    if (!showRefused(e.message)) $("err").textContent = e.message;
  }
  $("go").textContent = "Read letter";
  $("go").disabled = false;
};


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

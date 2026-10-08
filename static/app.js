// Persian letter reader - page behaviour.
// Photo in (file, drop or camera) -> optional drag-selection -> POST /predict -> show the answer.

const $ = id => document.getElementById(id);

let current = null;            // the photo (File) that will be read
let stream = null;             // live camera stream, while the camera is on
let mirrored = false;          // mirror the camera PREVIEW only (never the saved photo)
let facing = "environment";    // which camera to ask for: "environment" (rear) or "user" (front)
let sel = null;                // box dragged on the photo, in the photo's own pixels


// ---------- showing and hiding parts of the page ----------

function hideResults() {
  $("result").hidden = true;
  $("howread").hidden = true;
  $("savebox").hidden = true;
}

function showEmpty() {
  const hint = document.createElement("span");
  hint.id = "hint";
  hint.innerHTML = '<span class="big-hint" aria-hidden="true">&#8681;</span>Drop an image here, or click to choose one';
  $("drop").replaceChildren(hint);
  $("selrow").hidden = true;
  $("go").disabled = true;
}


// ---------- choosing a photo ----------

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


// ---------- drag a box on the photo to choose one letter ----------

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
    $("selhint").textContent = "Only the green box will be read.";
    hideResults();
  });
  img.addEventListener("click", e => e.stopPropagation());
}

$("clearSel").onclick = () => { clearSel(); hideResults(); };
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && sel) { clearSel(); hideResults(); }
});


// ---------- camera ----------

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
  $("camBtn").textContent = "Take photo";
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

$("camBtn").onclick = async () => {
  if (stream) { captureFrame(); return; }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    $("err").textContent = "Live camera needs https or localhost. Using the file picker instead.";
    $("cam").click();
    return;
  }
  try {
    await openStream();
    current = null;
    $("camBtn").textContent = "Capture";
    showCamControls(true);
    $("go").disabled = true;
    $("err").textContent = "";
    hideResults();
  } catch (e) {
    stream = null;
    showCamControls(false);
    $("err").textContent = (e && e.name === "NotAllowedError" ? "Camera permission was denied."
                                                              : "Camera not available.") + " Choose a file instead.";
    $("cam").click();
  }
};

$("camOff").onclick = () => { stopCam(); showEmpty(); };

$("mirrorBtn").onclick = () => { mirrored = !mirrored; applyMirror(); };

$("flipBtn").onclick = async () => {
  facing = facing === "user" ? "environment" : "user";
  try { await openStream(); } catch (e) { $("err").textContent = "Could not switch camera."; }
};


// ---------- reading the letter ----------

function showAnswer(d) {
  const [t, ...rest] = d.results;

  $("letter").textContent = t.letter;
  const span = document.createElement("span");
  span.textContent = t.latin;
  $("latin").replaceChildren(span);
  $("latin").classList.toggle("symbol", !/[a-z]/i.test(t.latin));   // eyn = ʿ, alef = -
  $("latin").classList.toggle("high", t.latin === "ʿ");
  $("name").textContent = t.name + (t.ascii !== t.latin ? " · plain: " + t.ascii : "");

  const pct = Math.round(t.confidence * 100);
  $("conf").textContent = "Confidence " + pct + "%" + (t.confidence < 0.6 ? " (low: retake the photo)" : "");
  $("confbar").firstElementChild.style.width = pct + "%";
  $("confbar").classList.toggle("low", t.confidence < 0.6);

  $("alts").replaceChildren(...rest.map(x => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = x.letter + " " + x.latin + " (" + x.name + ") " + Math.round(x.confidence * 100) + "%";
    return chip;
  }));
  $("how").textContent = "Read: " + d.how;

  $("found").src = d.found;
  $("view").src = d.view;
  $("truth").value = t.id;
  $("saved").textContent = "";
  $("saveBtn").disabled = false;

  $("result").hidden = false;
  $("howread").hidden = false;
  $("savebox").hidden = false;
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
    const r = await fetch("/predict", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Something went wrong.");
    showAnswer(d);
  } catch (e) {
    $("err").textContent = e.message;
  }
  $("go").textContent = "Read letter";
  $("go").disabled = false;
};


// ---------- save as test photo ----------

$("saveBtn").onclick = async () => {
  $("saveBtn").disabled = true;
  $("saved").textContent = "Saving...";
  try {
    const fd = new FormData();
    fd.append("image", current);
    fd.append("class_id", $("truth").value);
    const r = await fetch("/save", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Could not save.");
    $("saved").textContent = "Saved as " + d.folder + "/" + d.saved;
  } catch (e) {
    $("saved").textContent = "";
    $("err").textContent = e.message;
    $("saveBtn").disabled = false;
  }
};

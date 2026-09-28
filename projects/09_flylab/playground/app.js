import { Body, Fly, PRETTY, TASKS, antennae, emptyStim, parseNpy, rng, sampleArena, score,
  setTurn, targetOf, wilson, wrap, DT, N_HID } from "./flybrain.js";

const $ = s => document.querySelector(s);
const canvas = $("#arena"), ctx = canvas.getContext("2d");
const DEG = Math.PI / 180;

// What each brain has to do, and how it is judged (tasks.py).
const SPEC = {
  walk_forward: ["Goal compass only.", "Walk to the flag straight ahead and stop: within 1.5 mm, moving under 1.5 mm/s at the end."],
  turn: ["Goal compass only.", "Face the dashed direction: within 20°, without drifting more than 6 mm."],
  goto: ["Goal compass only.", "Walk to the flag and stop: within 1.5 mm, moving under 1.5 mm/s at the end."],
  odor_seek: ["Two antennae only; no compass.", "Find the food by smell and stop on it: within 1.5 mm, under 1.5 mm/s."],
  odor_avoid: ["Two antennae only; no compass.", "Get 6 mm further from the repellent without ever coming more than 0.5 mm closer."],
  light_seek: ["Two eyes only; no compass.", "Walk under the lamp and stop: within 1.5 mm, under 1.5 mm/s."],
  light_avoid: ["Two eyes only; no compass.", "Get 6 mm further from the lamp without ever coming more than 0.5 mm closer."],
};
const TARGET_NAME = { walk_forward: "flag", goto: "flag", odor_seek: "food", odor_avoid: "repellent", light_seek: "lamp", light_avoid: "lamp" };
const SENSE_ROWS = [
  ["smell strength", 0, 0, 2], ["smell left − right", 1, -2, 2], ["smell rising", 2, -3, 3],
  ["light strength", 3, 0, 2], ["light left − right", 4, -2, 2], ["light rising", 5, -3, 3],
  ["goal bearing (sin)", 6, -1, 1], ["goal bearing (cos)", 7, -1, 1], ["goal distance", 8, 0, 1], ["has a goal", 9, 0, 1],
];
// training.status -> label, as on the dashboard (train_brains.py LEVELS)
const LEVELS = [{ val: 6, thr: 1 }, { val: 12, thr: 1 }, { val: 24, thr: 0.95 }];

const S = {
  body: null, brains: {}, status: {}, videos: {}, task: "odor_seek", arena: null, fly: null, trail: [],
  running: false, lesion: {}, silenced: new Set(), noise: false, speed: 1, view: null, runs: 0,
  drag: null, last: null, acc: 0,
};

// ------------------------------------------------------------------ arenas
function defaultArena(task) {
  const a = { task, x0: 0, y0: 0, th0: 0, stim: emptyStim() }, s = a.stim;
  const at = (r, deg) => [r * Math.cos(deg * DEG), r * Math.sin(deg * DEG)];
  if (task === "walk_forward") { a.distance = 12; placeWalkGoal(a); }
  else if (task === "turn") setTurn(a, 90 * DEG);
  else if (task === "goto") { s.goal_on = 1; s.goal_x = 10; s.goal_y = -6; }
  else if (task === "odor_seek") { s.odor_on = 1; [s.odor_x, s.odor_y] = at(11, 60); }
  else if (task === "odor_avoid") { s.odor_on = 1; [s.odor_x, s.odor_y] = at(3.5, 60); }
  else if (task === "light_seek") { s.light_on = 1; [s.light_x, s.light_y] = at(12, -70); }
  else { s.light_on = 1; [s.light_x, s.light_y] = at(5, -70); }
  return a;
}
function placeWalkGoal(a) {            // walk_forward's flag sits `distance` mm along the start heading
  a.stim.goal_on = 1;
  a.stim.goal_x = a.x0 + a.distance * Math.cos(a.th0);
  a.stim.goal_y = a.y0 + a.distance * Math.sin(a.th0);
}
function setTarget(a, x, y) {
  const s = a.stim;
  if (s.odor_on) { s.odor_x = x; s.odor_y = y; }
  else if (s.light_on) { s.light_x = x; s.light_y = y; }
  else if (a.task === "goto") { s.goal_x = x; s.goal_y = y; }
}

// ------------------------------------------------------------------- view
function fitView() {
  const a = S.arena, pts = [[a.x0, a.y0]], t = targetOf(a);
  if (t) pts.push(t);
  if (S.fly) S.trail.forEach(p => pts.push(p));
  let cx = pts.reduce((s, p) => s + p[0], 0) / pts.length, cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  if (a.task.endsWith("avoid")) { cx = t[0]; cy = t[1]; }
  let span = 26;
  pts.forEach(p => { span = Math.max(span, 2 * Math.max(Math.abs(p[0] - cx), Math.abs(p[1] - cy)) + 10); });
  S.view = { cx, cy, span };
}
const W = () => canvas.width;
const toPx = (x, y) => [(x - S.view.cx) / S.view.span * W() + W() / 2, W() / 2 - (y - S.view.cy) / S.view.span * W()];
const toMm = (px, py) => [(px - W() / 2) / W() * S.view.span + S.view.cx, (W() / 2 - py) / W() * S.view.span + S.view.cy];
const mm = d => d / S.view.span * W();

// ------------------------------------------------------------------ drawing
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function draw() {
  const a = S.arena, s = a.stim, w = W();
  ctx.clearRect(0, 0, w, w);
  // stimulus field
  const glow = (x, y, rgb, reach) => {
    const [px, py] = toPx(x, y), g = ctx.createRadialGradient(px, py, 0, px, py, mm(reach));
    [0, 0.1, 0.3, 0.6, 1].forEach(f => g.addColorStop(f, `rgba(${rgb},${0.42 * Math.exp(-3 * f)})`));
    ctx.fillStyle = g; ctx.fillRect(0, 0, w, w);
  };
  if (s.odor_on) glow(s.odor_x, s.odor_y, a.task === "odor_seek" ? "86,160,40" : "140,60,170", 30);
  if (s.light_on) glow(s.light_x, s.light_y, "240,185,40", 36);
  // grid, 5 mm
  ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
  const x0 = Math.floor((S.view.cx - S.view.span / 2) / 5) * 5, y0 = Math.floor((S.view.cy - S.view.span / 2) / 5) * 5;
  for (let g = 0; g <= S.view.span + 5; g += 5) {
    const [gx] = toPx(x0 + g, 0), [, gy] = toPx(0, y0 + g);
    ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, w); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
  }
  // scale bar
  ctx.fillStyle = css("--muted"); ctx.strokeStyle = css("--muted"); ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(16, w - 16); ctx.lineTo(16 + mm(5), w - 16); ctx.stroke();
  ctx.font = "12px system-ui"; ctx.fillText("5 mm", 16, w - 22);
  drawTarget();
  // trail
  if (S.trail.length > 1) {
    ctx.strokeStyle = css("--s1"); ctx.lineWidth = 2; ctx.beginPath();
    S.trail.forEach((p, i) => { const [px, py] = toPx(p[0], p[1]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
    ctx.stroke();
  }
  const f = S.fly;
  const pose = f ? [f.x, f.y, f.th] : [a.x0, a.y0, a.th0];
  if (!S.running && !f) drawHandle(pose);
  drawFly(...pose);
}

function drawTarget() {
  const a = S.arena, s = a.stim, t = targetOf(a);
  const text = css("--text");
  if (a.task === "turn") {
    const [px, py] = toPx(a.x0, a.y0), h = a.targetHeading, L = mm(7);
    ctx.setLineDash([6, 5]); ctx.strokeStyle = "#c0392b"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px + L * Math.cos(h), py - L * Math.sin(h)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#c0392b"; ctx.beginPath(); ctx.arc(px + L * Math.cos(h), py - L * Math.sin(h), 7, 0, 7); ctx.fill();
    return;
  }
  const [px, py] = toPx(t[0], t[1]);
  if (!a.task.endsWith("avoid")) {        // success radius
    ctx.setLineDash([4, 4]); ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(px, py, mm(1.5), 0, 7); ctx.stroke(); ctx.setLineDash([]);
  }
  if (s.odor_on) {
    ctx.fillStyle = a.task === "odor_seek" ? "#9ccc2e" : "#8c3caa";
    ctx.beginPath(); ctx.arc(px, py, Math.max(6, mm(0.9)), 0, 7); ctx.fill();
  } else if (s.light_on) {
    ctx.strokeStyle = "#e0a800"; ctx.lineWidth = 2;
    for (let k = 0; k < 8; k++) {
      const ang = k * Math.PI / 4;
      ctx.beginPath(); ctx.moveTo(px + 11 * Math.cos(ang), py + 11 * Math.sin(ang));
      ctx.lineTo(px + 16 * Math.cos(ang), py + 16 * Math.sin(ang)); ctx.stroke();
    }
    ctx.fillStyle = "#f5c518"; ctx.beginPath(); ctx.arc(px, py, 8, 0, 7); ctx.fill();
  } else {
    ctx.strokeStyle = text; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px, py - 22); ctx.stroke();
    ctx.fillStyle = "#c0392b"; ctx.beginPath(); ctx.moveTo(px, py - 22); ctx.lineTo(px + 13, py - 17); ctx.lineTo(px, py - 12); ctx.fill();
  }
}

function drawHandle([x, y, th]) {
  const [px, py] = toPx(x, y), L = mm(4.5);
  ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px + L * Math.cos(th), py - L * Math.sin(th)); ctx.stroke();
  ctx.fillStyle = css("--surface"); ctx.beginPath(); ctx.arc(px + L * Math.cos(th), py - L * Math.sin(th), 6, 0, 7);
  ctx.fill(); ctx.stroke();
}

function drawFly(x, y, th) {
  // body sized to the real fly: ~2.5 mm long, drawn in world units
  const [px, py] = toPx(x, y), k = mm(1);
  ctx.save(); ctx.translate(px, py); ctx.rotate(-th);
  ctx.fillStyle = "rgba(170,200,235,0.55)";
  [1, -1].forEach(sd => { ctx.beginPath(); ctx.ellipse(-0.55 * k, sd * 0.45 * k, 0.95 * k, 0.32 * k, sd * 0.35, 0, 7); ctx.fill(); });
  ctx.fillStyle = "#a0662a";
  ctx.beginPath(); ctx.ellipse(-0.8 * k, 0, 0.75 * k, 0.42 * k, 0, 0, 7); ctx.fill();   // abdomen
  ctx.beginPath(); ctx.ellipse(0.05 * k, 0, 0.42 * k, 0.36 * k, 0, 0, 7); ctx.fill();   // thorax
  ctx.fillStyle = "#b8402f"; ctx.beginPath(); ctx.arc(0.55 * k, 0, 0.24 * k, 0, 7); ctx.fill(); // head
  ctx.restore();
  // antennae where the model senses odor, coloured by what each one smells
  const a = S.arena;
  antennae(x, y, th).forEach(([ax, ay], i) => {
    const [qx, qy] = toPx(ax, ay);
    const side = i ? "rightAntenna" : "leftAntenna";
    let o = a.stim.odor_on ? Math.exp(-Math.hypot(ax - a.stim.odor_x, ay - a.stim.odor_y) / 10) : 0;
    if (S.lesion[side]) o = 0;
    ctx.fillStyle = S.lesion[side] ? css("--muted") : `rgba(40,120,20,${0.35 + 0.65 * o})`;
    ctx.beginPath(); ctx.arc(qx, qy, Math.max(2.5, 0.09 * k), 0, 7); ctx.fill();
  });
}

// ------------------------------------------------------------------ panels
function bars(el, rows, values, cls) {
  el.className = "bars" + (cls ? " " + cls : "");
  el.innerHTML = rows.map(([label, , lo, hi], i) => {
    const v = values ? values[i] : 0, span = hi - lo, z = lo < 0 ? (0 - lo) / span : 0;
    const f = Math.min(1, Math.max(0, (v - lo) / span));
    const left = Math.min(z, f) * 100, width = Math.abs(f - z) * 100;
    const off = values && Math.abs(v) < 1e-9 ? " off" : "";
    return `<span class="lbl${off}">${label}</span><span class="track${off}"><i style="left:${left}%;width:${width}%"></i>${lo < 0 ? "<b></b>" : ""}</span><span class="val${off}">${values ? v.toFixed(2) : "–"}</span>`;
  }).join("");
}
function showState(out, feats) {
  bars($("#senses"), SENSE_ROWS, feats ? SENSE_ROWS.map(r => feats[r[1]]) : null);
  bars($("#drives"), [["left", 0, -0.5, 1.2], ["right", 1, -0.5, 1.2]], out ? out.drives : null, "drives");
  const btns = $("#hidden").children;
  for (let j = 0; j < N_HID; j++) {
    const h = out ? out.hidden[j] : 0, b = btns[j];
    b.style.background = h >= 0 ? `rgba(42,120,214,${Math.abs(h)})` : `rgba(235,104,52,${Math.abs(h)})`;
    b.title = `neuron ${j + 1}: ${h.toFixed(2)}${S.silenced.has(j) ? " (silenced)" : ""}`;
    b.classList.toggle("silenced", S.silenced.has(j));
    b.textContent = S.silenced.has(j) ? "×" : "";
  }
}

function showRecord() {
  const st = S.status[S.task], el = $("#record"), [uses, rule] = SPEC[S.task];
  if (!st) { el.innerHTML = `<div class="dim">No training record found.</div><p>${uses} ${rule}</p>`; return; }
  const level = typeof st.level === "number" ? st.level : 0, L = LEVELS[Math.min(level, 2)];
  const b = st.best || {}, t = st.test || {}, nb = b.n || (b.physics_errors || []).length;
  const passed = b.physics_val >= L.thr - 1e-9 && b.surrogate >= 0.95 && nb >= L.val;
  const optimal = level >= 2 && passed && t.physics_test >= 0.9 - 1e-9;
  const badge = optimal ? ["optimal", "optimal"] : passed ? ["passed", `level ${level + 1} of 3 · passed`] : ["training", `level ${level + 1} of 3 · still training`];
  const tn = t.n || (t.physics_errors || []).length, tk = Math.round((t.physics_test || 0) * tn);
  const [lo, hi] = wilson(tk, tn);
  el.innerHTML = `
    <div class="line"><span class="big">${tk}/${tn}</span><span>held-out physics arenas passed</span>
      <span class="dim">95% CI ${Math.round(100 * lo)}–${Math.round(100 * hi)}%</span></div>
    <div class="line"><span class="badge ${badge[0]}">${badge[1]}</span>
      <span class="dim">brain from round ${b.round ?? "?"} · ${(st.generations || 0).toLocaleString()} ES generations · ${(st.physics_runs || 0).toLocaleString()} physics runs</span></div>
    <p style="margin:8px 0 0">${uses} ${rule}</p>`;
}

function showFilm() {
  const v = S.videos[S.task], film = $("#film");
  film.src = `../dashboard/videos/${S.task}.mp4`;
  film.poster = `../dashboard/videos/${S.task}.png`;
  $("#film-cap").textContent = v ? `Brain from round ${v.brain_round} on an arena it never trained on (${v.arena}): ${v.outcome}.` : "";
}

function showDownloads() {
  $("#downloads").innerHTML = TASKS.map(t =>
    `<li>${PRETTY[t]}: <a href="../state/brains/${t}.npy" download>${t}.npy</a> · <a href="#" data-json="${t}">${t}.json</a></li>`).join("") +
    `<li>Calibrated body: <a href="calibration.json" download>calibration.json</a> (speeds and turn rates measured in MuJoCo)</li>
     <li>Training history per brain: <a href="https://github.com/ROHITCRAFTSYT/FruitFlyBrain/tree/main/projects/09_flylab/training">training/&lt;skill&gt;/</a></li>`;
}

// -------------------------------------------------------------- simulation
async function brain(task) {
  if (!S.brains[task]) {
    const r = await fetch(`../state/brains/${task}.npy`, { cache: "no-cache" });
    if (!r.ok) throw new Error(`could not load ${task}.npy (${r.status})`);
    S.brains[task] = parseNpy(await r.arrayBuffer());
  }
  return S.brains[task];
}

function resetRun() {
  S.running = false; S.fly = null; S.trail = []; S.acc = 0;
  $("#run").textContent = "Run"; $("#result").hidden = true;
  fitView(); showState(null, null); draw();
}

async function start() {
  if (!S.fly || S.fly.done) {
    const theta = await brain(S.task);
    S.fly = new Fly(theta, S.body, structuredClone(S.arena),
      { noise: S.noise, seed: 1 + S.runs++, lesion: { ...S.lesion }, silenced: new Set(S.silenced) });
    S.trail = [[S.fly.x, S.fly.y]];
    $("#result").hidden = true;
  }
  S.running = true; $("#run").textContent = "Pause"; S.last = null;
  requestAnimationFrame(tick);
}

function tick(ts) {
  if (!S.running) return;
  const dt = S.last == null ? 0 : Math.min(0.1, (ts - S.last) / 1000);
  S.last = ts; S.acc += dt * S.speed;
  let out = null;
  while (S.acc >= DT && !S.fly.done) {
    out = S.fly.step(); S.acc -= DT;
    S.trail.push([S.fly.x, S.fly.y]);
  }
  const f = S.fly, n = f.rec.drives.length;
  if (n) showState({ drives: f.rec.drives[n - 1], hidden: f.rec.hidden[n - 1] }, f.rec.feats[n - 1]);
  const [px, py] = toPx(f.x, f.y);
  if (px < 30 || py < 30 || px > W() - 30 || py > W() - 30) fitView();
  draw();
  if (f.done) finish(); else requestAnimationFrame(tick);
}

function finish() {
  S.running = false; $("#run").textContent = "Run again";
  const f = S.fly, r = score(f.arena, f.rec.x, f.rec.y, f.rec.th), el = $("#result");
  const task = S.task, name = TARGET_NAME[task];
  let msg;
  if (task === "turn") msg = `ended ${r.headingErrorDeg.toFixed(0)}° from the target heading and drifted ${r.driftMm.toFixed(1)} mm (needs under 20° and 6 mm)`;
  else if (task.endsWith("avoid")) msg = `went from ${r.startMm.toFixed(1)} to ${r.endMm.toFixed(1)} mm from the ${name}` +
    (r.approachMm > 0.05 ? `, after first coming ${r.approachMm.toFixed(1)} mm closer` : "") + " (needs +6 mm, never more than 0.5 mm closer)";
  else msg = `ended ${r.endMm.toFixed(1)} mm from the ${name}, moving ${r.finalSpeed.toFixed(1)} mm/s (needs under 1.5 mm and 1.5 mm/s)`;
  const mods = [...Object.keys(S.lesion).filter(k => S.lesion[k]).map(k => k.replace(/([A-Z])/g, " $1").toLowerCase() + " removed"),
    S.silenced.size ? `${S.silenced.size} neuron${S.silenced.size > 1 ? "s" : ""} silenced` : "", S.noise ? "training noise on" : ""].filter(Boolean);
  el.className = "result " + (r.success ? "pass" : "fail");
  el.textContent = `${r.success ? "Pass" : "Fail"}: ${msg}.${mods.length ? " With " + mods.join(", ") + "." : ""}`;
  el.hidden = false;
}

async function batch() {
  const btn = $("#batch"), out = $("#batch-out"), n = 200;
  btn.disabled = true; out.textContent = "running…";
  const theta = await brain(S.task), R = rng(20260928), task = S.task;
  let k = 0, i = 0;
  const lesion = { ...S.lesion }, silenced = new Set(S.silenced);
  await new Promise(done => {
    const chunk = () => {
      for (let j = 0; j < 25 && i < n; j++, i++)
        k += new Fly(theta, S.body, sampleArena(task, R), { noise: true, seed: 5000 + i, lesion, silenced }).run().success;
      out.textContent = `${i}/${n}…`;
      i < n ? setTimeout(chunk, 0) : done();
    };
    chunk();
  });
  const [lo, hi] = wilson(k, n), sur = S.status[task]?.best?.surrogate;
  out.textContent = `${k}/${n} passed (95% CI ${Math.round(100 * lo)}–${Math.round(100 * hi)}%)` +
    (sur != null ? `. Training measured ${Math.round(100 * sur)}% for this brain on the same kind of arenas, without lesions.` : "");
  btn.disabled = false;
}

// ------------------------------------------------------------ interaction
function pick(px, py) {
  const a = S.arena, [mx, my] = toMm(px, py), near = (x, y, r) => Math.hypot(mx - x, my - y) < r;
  const tol = S.view.span / 40;
  if (a.task === "turn") {
    const h = a.targetHeading;
    if (near(a.x0 + 7 * Math.cos(h), a.y0 + 7 * Math.sin(h), 1.2 * tol)) return "turnTarget";
  }
  if (near(a.x0 + 4.5 * Math.cos(a.th0), a.y0 + 4.5 * Math.sin(a.th0), 1.2 * tol)) return "heading";
  const t = targetOf(a);
  if (t && near(t[0], t[1], 1.6 * tol)) return "target";
  if (near(a.x0, a.y0, 1.6 * tol)) return "fly";
  return null;
}
function onDrag(kind, mx, my) {
  const a = S.arena;
  if (kind === "fly") { a.x0 = mx; a.y0 = my; }
  else if (kind === "heading") a.th0 = Math.atan2(my - a.y0, mx - a.x0);
  else if (kind === "turnTarget") setTurn(a, Math.atan2(my - a.y0, mx - a.x0));
  else if (kind === "target") {
    if (a.task === "walk_forward") {
      a.distance = Math.max(2, Math.min(25, (mx - a.x0) * Math.cos(a.th0) + (my - a.y0) * Math.sin(a.th0)));
    } else setTarget(a, mx, my);
  }
  if (a.task === "walk_forward") placeWalkGoal(a);
  if (a.task === "turn" && kind === "fly") setTurn(a, a.targetHeading);
  S.fly = null; S.trail = []; $("#result").hidden = true;
  draw();
}
function eventMm(e) {
  const r = canvas.getBoundingClientRect(), sc = canvas.width / r.width;
  return [(e.clientX - r.left) * sc, (e.clientY - r.top) * sc];
}
canvas.addEventListener("pointerdown", e => {
  if (S.running) return;
  const [px, py] = eventMm(e), k = pick(px, py);
  if (!k) return;
  S.drag = k; canvas.setPointerCapture(e.pointerId); canvas.classList.add("dragging");
});
canvas.addEventListener("pointermove", e => {
  const [px, py] = eventMm(e);
  if (S.drag) { onDrag(S.drag, ...toMm(px, py)); return; }
  canvas.style.cursor = !S.running && pick(px, py) ? "grab" : "default";
});
canvas.addEventListener("pointerup", () => {
  if (!S.drag) return;
  S.drag = null; canvas.classList.remove("dragging"); fitView(); draw();
});

function selectTask(task) {
  S.task = task; S.arena = defaultArena(task); S.silenced.clear();
  document.querySelectorAll("#skills button").forEach(b => b.setAttribute("aria-selected", b.dataset.task === task));
  history.replaceState(null, "", `?skill=${task}`);
  $("#hint").textContent = task === "turn"
    ? "Drag the fly to move it, its round handle to rotate it, and the red dot to set the direction to face."
    : `Drag the fly to move it, its round handle to rotate it, and the ${TARGET_NAME[task]} to move it.`;
  $("#batch-out").textContent = "";
  showRecord(); showFilm(); resetRun();
  brain(task).catch(fail);
}

function fail(err) { const e = $("#error"); e.hidden = false; e.textContent = String(err.message || err); }

// ------------------------------------------------------------------- boot
async function boot() {
  $("#skills").innerHTML = TASKS.map(t => `<button type="button" role="tab" data-task="${t}">${PRETTY[t]}</button>`).join("");
  $("#skills").addEventListener("click", e => { const b = e.target.closest("button"); if (b) selectTask(b.dataset.task); });
  $("#hidden").innerHTML = Array.from({ length: N_HID }, (_, j) => `<button type="button" data-n="${j}" aria-label="neuron ${j + 1}"></button>`).join("");
  $("#hidden").addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    const j = +b.dataset.n; S.silenced.has(j) ? S.silenced.delete(j) : S.silenced.add(j);
    if (S.fly) { S.fly.silenced = new Set(S.silenced); }
    showState(S.fly && S.fly.rec.drives.length ? { drives: S.fly.rec.drives.at(-1), hidden: S.fly.rec.hidden.at(-1) } : null,
      S.fly && S.fly.rec.feats.length ? S.fly.rec.feats.at(-1) : null);
  });
  document.querySelectorAll("[data-lesion]").forEach(cb => cb.addEventListener("change", () => {
    S.lesion[cb.dataset.lesion] = cb.checked; if (S.fly) S.fly.lesion = { ...S.lesion }; draw();
  }));
  $("#run").addEventListener("click", () => S.running ? (S.running = false, $("#run").textContent = "Resume") : start().catch(fail));
  $("#reset").addEventListener("click", resetRun);
  $("#random").addEventListener("click", () => {
    const a = sampleArena(S.task, rng((Date.now() ^ (S.runs++ * 7919)) >>> 0));
    if (S.task === "walk_forward") { a.distance = Math.hypot(a.stim.goal_x, a.stim.goal_y); }
    if (S.task === "turn") a.targetHeading = wrap(Math.atan2(a.stim.goal_y, a.stim.goal_x));
    S.arena = a; resetRun();
  });
  $("#speed").addEventListener("change", e => { S.speed = +e.target.value; });
  $("#noise").addEventListener("change", e => { S.noise = e.target.checked; });
  $("#batch").addEventListener("click", () => batch().catch(fail));
  $("#downloads").addEventListener("click", async e => {
    const t = e.target.dataset?.json; if (!t) return;
    e.preventDefault();
    const theta = Array.from(await brain(t)), st = S.status[t] || {};
    const blob = new Blob([JSON.stringify({ skill: t, architecture: "17 -> 32 tanh -> 2, d = 0.35 + 0.85 tanh(o), gate 0.02-0.10",
      layout: { W1: [17, 32], b1: [32], W2: [32, 2], b2: [2] }, round: st.best?.round,
      heldOutPhysicsTest: st.test?.physics_test, heldOutArenas: st.test?.n, theta }, null, 1)], { type: "application/json" });
    const url = URL.createObjectURL(blob), link = Object.assign(document.createElement("a"), { href: url, download: `${t}.json` });
    link.click(); URL.revokeObjectURL(url);
  });
  showDownloads();
  // resolution follows the element's on-screen size
  new ResizeObserver(() => {
    const px = Math.round(Math.min(900, canvas.getBoundingClientRect().width * (window.devicePixelRatio || 1)));
    if (px > 0 && px !== canvas.width) { canvas.width = canvas.height = px; if (S.view) draw(); }
  }).observe(canvas);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => S.view && draw());

  const cal = await fetch("calibration.json").then(r => { if (!r.ok) throw new Error("calibration.json missing"); return r.json(); });
  S.body = new Body(cal);
  await Promise.all(TASKS.map(t => fetch(`../training/${t}/status.json`, { cache: "no-cache" })
    .then(r => r.ok ? r.json() : null).then(j => { S.status[t] = j; }).catch(() => {})));
  await fetch("../dashboard/videos/index.json", { cache: "no-cache" }).then(r => r.ok ? r.json() : { videos: [] })
    .then(j => (j.videos || []).forEach(v => { S.videos[v.skill] = v; })).catch(() => {});
  const want = new URLSearchParams(location.search).get("skill");
  selectTask(TASKS.includes(want) ? want : "odor_seek");
}

boot().catch(err => fail(location.protocol === "file:"
  ? "Open this page through a web server (browsers block file:// fetches), e.g. python -m http.server in the FruitFlyBrain folder."
  : err));

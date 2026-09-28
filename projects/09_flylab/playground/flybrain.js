// FlyLab brain engine for the browser (and Node, for the parity tests).
//
// A line-by-line port of the Python used in training:
//   world.py      senses (antennae, eyes, goal compass, sensory integration)
//   brain.py      the 17 -> 32 -> 2 network and its motor gate
//   surrogate.py  the calibrated body (drive -> speed / slip / turn, CPG lag)
//   tasks.py      arena sampling and the pass/fail rules
// tools/check_core.mjs checks it against Python to ~1e-12 on every brain.

export const TASKS = ["walk_forward", "turn", "goto", "odor_seek", "odor_avoid", "light_seek", "light_avoid"];
export const PRETTY = {
  walk_forward: "Walk forward", turn: "Turn in place", goto: "Go to a location",
  odor_seek: "Find food by smell", odor_avoid: "Escape a bad smell",
  light_seek: "Walk toward light", light_avoid: "Hide from light",
};
export const DURATION_S = {
  walk_forward: 5, turn: 3, goto: 7, odor_seek: 8, odor_avoid: 4, light_seek: 7, light_avoid: 4,
};
export const SENSE_NAMES = ["odor", "odor L-R", "odor dt", "light", "light L-R", "light dt",
  "goal sin", "goal cos", "goal dist", "goal known"];

// world.py
const ANTENNA_FORWARD = 0.52, ANTENNA_HALF_SPAN = 0.19;
const EYE_AXIS = 30 * Math.PI / 180;
const ODOR_DECAY_MM = 10, LIGHT_SCALE_MM = 30, LIGHT_HEIGHT_MM = 1;
const SENSE_BETA = 0.4;
// brain.py
export const N_SENSES = 10, N_TASKS = 7, N_IN = 17, N_HID = 32, N_OUT = 2;
export const N_PARAMS = N_IN * N_HID + N_HID + N_HID * N_OUT + N_OUT;   // 642
const DRIVE_CENTER = 0.35, DRIVE_SPAN = 0.85, GATE_LO = 0.02, GATE_HI = 0.10;
// surrogate.py
export const DT = 0.05;
const DR_GAIN = 0.2, DR_VEER = 0.25, DR_JITTER_MM = 0.15, DR_YAW_WOBBLE = 4 * Math.PI / 180, STRIDE_DRIVE = 0.6;
// tasks.py
const SUCCESS_RADIUS_MM = 1.5, STOPPED_MM_S = 1.5, STOP_WINDOW_S = 0.5;
const AVOID_GAIN_MM = 6, AVOID_SLACK_MM = 0.5, TURN_MAX_DISP_MM = 6;

const TAU = 2 * Math.PI;
export const wrap = a => ((((a + Math.PI) % TAU) + TAU) % TAU) - Math.PI;
const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// ---------------------------------------------------------------- loading
/** Parse a NumPy .npy file holding a little-endian float64 array. */
export function parseNpy(buf) {
  const bytes = new Uint8Array(buf);
  if (bytes[0] !== 0x93 || String.fromCharCode(...bytes.slice(1, 6)) !== "NUMPY") throw new Error("not a .npy file");
  const major = bytes[6];
  const hlen = major === 1 ? bytes[8] | (bytes[9] << 8) : new DataView(buf).getUint32(8, true);
  const start = (major === 1 ? 10 : 12) + hlen;
  const header = new TextDecoder().decode(bytes.slice(major === 1 ? 10 : 12, start));
  if (!header.includes("'<f8'")) throw new Error("expected float64 weights, got " + header);
  return new Float64Array(buf.slice(start));
}

/** The calibrated body: velocities measured in MuJoCo for a grid of drives. */
export class Body {
  constructor(cal) {
    this.grid = cal.grid; this.vFwd = cal.v_fwd; this.vLat = cal.v_lat; this.omega = cal.omega; this.tau = cal.tau;
  }
  _index(v) {                      // np.interp(v, grid, arange(n)), clamped
    const g = this.grid, n = g.length;
    if (v <= g[0]) return 0;
    if (v >= g[n - 1]) return n - 1;
    let k = 0;
    while (v >= g[k + 1]) k++;
    return k + (v - g[k]) / (g[k + 1] - g[k]);
  }
  _interp(t, fi, fj) {
    const n = this.grid.length;
    const i0 = clip(Math.floor(fi), 0, n - 2), j0 = clip(Math.floor(fj), 0, n - 2);
    const a = fi - i0, b = fj - j0;
    return (1 - a) * (1 - b) * t[i0][j0] + a * (1 - b) * t[i0 + 1][j0]
      + (1 - a) * b * t[i0][j0 + 1] + a * b * t[i0 + 1][j0 + 1];
  }
  velocities(dl, dr) {
    const fi = this._index(dl), fj = this._index(dr);
    return [this._interp(this.vFwd, fi, fj), this._interp(this.vLat, fi, fj), this._interp(this.omega, fi, fj)];
  }
}

// ----------------------------------------------------------------- senses
export function antennae(x, y, th) {
  const c = Math.cos(th), s = Math.sin(th);
  const hx = x + ANTENNA_FORWARD * c, hy = y + ANTENNA_FORWARD * s;
  return [[hx - ANTENNA_HALF_SPAN * s, hy + ANTENNA_HALF_SPAN * c], [hx + ANTENNA_HALF_SPAN * s, hy - ANTENNA_HALF_SPAN * c]];
}
const odorAt = (px, py, sx, sy) => Math.exp(-Math.hypot(px - sx, py - sy) / ODOR_DECAY_MM);
const lightAt = (px, py, sx, sy) => 1 / (1 + (Math.hypot(px - sx, py - sy) / LIGHT_SCALE_MM) ** 2);

/**
 * world.sense for one fly. `stim` has odor_on/x/y, light_on/x/y, goal_on/x/y,
 * goal_heading_only. `lesion` can silence one antenna or one eye.
 * Returns {feats, state} (state carries the sensory integrator).
 */
export function sense(x, y, th, stim, prev, lesion = {}) {
  const [[lx, ly], [rx, ry]] = antennae(x, y, th);
  let oL = odorAt(lx, ly, stim.odor_x, stim.odor_y) * stim.odor_on;
  let oR = odorAt(rx, ry, stim.odor_x, stim.odor_y) * stim.odor_on;
  if (lesion.leftAntenna) oL = 0;
  if (lesion.rightAntenna) oR = 0;
  const oMean = 0.5 * (oL + oR), oCon = (oL - oR) / (oL + oR + 1e-9);

  const I = lightAt(x, y, stim.light_x, stim.light_y) * stim.light_on;
  const bear = wrap(Math.atan2(stim.light_y - y, stim.light_x - x) - th);
  const r = Math.hypot(stim.light_x - x, stim.light_y - y);
  const lateral = r / Math.hypot(r, LIGHT_HEIGHT_MM);
  const overhead = Math.cos(EYE_AXIS) * (1 - lateral);
  let vL = I * (lateral * Math.max(0, Math.cos(bear - EYE_AXIS)) + overhead);
  let vR = I * (lateral * Math.max(0, Math.cos(bear + EYE_AXIS)) + overhead);
  if (lesion.leftEye) vL = 0;
  if (lesion.rightEye) vR = 0;
  const vMean = 0.5 * (vL + vR), vCon = (vL - vR) / (vL + vR + 1e-6);

  const gb = wrap(Math.atan2(stim.goal_y - y, stim.goal_x - x) - th);
  const gd = Math.hypot(stim.goal_x - x, stim.goal_y - y) * (1 - (stim.goal_heading_only || 0));
  const gOn = stim.goal_on;
  const now = [oMean, oCon, vMean, vCon, Math.sin(gb) * gOn, Math.cos(gb) * gOn, Math.tanh(gd / 10) * gOn];
  let filt, dO = 0, dV = 0;
  if (!prev) filt = now;
  else {
    filt = now.map((v, i) => prev[i] + SENSE_BETA * (v - prev[i]));
    dO = filt[0] - prev[0]; dV = filt[2] - prev[2];
  }
  const feats = [filt[0] * 2, filt[1] * 20, clip(dO * 200, -3, 3), filt[2] * 2, filt[3] * 2,
    clip(dV * 200, -3, 3), filt[4], filt[5], filt[6], gOn];
  return { feats, state: filt };
}

// ------------------------------------------------------------------ brain
/** brain.act: returns drives [L, R] plus the hidden activity (for display). */
export function act(theta, feats, task, silenced = null) {
  const x = feats.concat(TASKS.map(t => (t === task ? 1 : 0)));
  const b1 = N_IN * N_HID, w2 = b1 + N_HID, b2 = w2 + N_HID * N_OUT;
  const h = new Array(N_HID);
  for (let j = 0; j < N_HID; j++) {
    let s = theta[b1 + j];
    for (let i = 0; i < N_IN; i++) s += x[i] * theta[i * N_HID + j];
    h[j] = silenced && silenced.has(j) ? 0 : Math.tanh(s);
  }
  const drives = [0, 0];
  for (let o = 0; o < N_OUT; o++) {
    let s = theta[b2 + o];
    for (let j = 0; j < N_HID; j++) s += h[j] * theta[w2 + j * N_OUT + o];
    const d = DRIVE_CENTER + DRIVE_SPAN * Math.tanh(s);
    drives[o] = d * clip((Math.abs(d) - GATE_LO) / (GATE_HI - GATE_LO), 0, 1);
  }
  return { drives, hidden: h };
}

// -------------------------------------------------------------- randomness
export function rng(seed) {                     // mulberry32 + Box-Muller
  let a = seed >>> 0;
  const uni = () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = (mu = 0, sd = 1) => mu + sd * Math.sqrt(-2 * Math.log(1 - uni())) * Math.cos(TAU * uni());
  return { uni, normal, uniform: (lo, hi) => lo + (hi - lo) * uni() };
}

// ----------------------------------------------------------------- arenas
export function emptyStim() {
  return { odor_on: 0, odor_x: 0, odor_y: 0, light_on: 0, light_x: 0, light_y: 0,
    goal_on: 0, goal_x: 0, goal_y: 0, goal_heading_only: 0 };
}

/** tasks.sample: a random arena. The fly starts at the origin, facing th0. */
export function sampleArena(task, R) {
  const th0 = R.uniform(-Math.PI, Math.PI);
  const s = emptyStim(), a = { task, x0: 0, y0: 0, th0, stim: s };
  const polar = (r, ang) => [r * Math.cos(ang), r * Math.sin(ang)];
  if (task === "walk_forward") {
    const D = R.uniform(5, 15);
    s.goal_on = 1; [s.goal_x, s.goal_y] = polar(D, th0);
  } else if (task === "turn") {
    const A = (R.uni() < 0.5 ? -1 : 1) * R.uniform(Math.PI / 4, Math.PI);
    setTurn(a, wrap(th0 + A));
  } else if (task === "goto") {
    s.goal_on = 1; [s.goal_x, s.goal_y] = polar(R.uniform(6, 16), R.uniform(-Math.PI, Math.PI));
  } else if (task.startsWith("odor")) {
    s.odor_on = 1; [s.odor_x, s.odor_y] = polar(task === "odor_seek" ? R.uniform(8, 15) : R.uniform(2.5, 5), R.uniform(-Math.PI, Math.PI));
  } else {
    s.light_on = 1; [s.light_x, s.light_y] = polar(task === "light_seek" ? R.uniform(8, 16) : R.uniform(4, 8), R.uniform(-Math.PI, Math.PI));
  }
  return a;
}

/** Point a turn arena at a target heading (tasks.py: a far goal, heading only). */
export function setTurn(arena, heading) {
  const s = arena.stim;
  s.goal_on = 1; s.goal_heading_only = 1;
  s.goal_x = 1000 * Math.cos(heading); s.goal_y = 1000 * Math.sin(heading);
  arena.targetHeading = heading;
}

/** Where the thing the fly cares about is (for drawing and scoring). */
export function targetOf(arena) {
  const s = arena.stim;
  if (s.odor_on) return [s.odor_x, s.odor_y];
  if (s.light_on) return [s.light_x, s.light_y];
  if (s.goal_on && !s.goal_heading_only) return [s.goal_x, s.goal_y];
  return null;
}

// -------------------------------------------------------------- simulation
/**
 * One fly in the calibrated surrogate body. `noise` switches on the same
 * domain randomization the brains were trained under (body gains, veer,
 * response lag, gait wobble on the senses, sensor and motor noise).
 */
export class Fly {
  constructor(theta, body, arena, { noise = false, seed = 1, lesion = {}, silenced = null } = {}) {
    Object.assign(this, { theta, body, arena, noise, lesion, silenced });
    this.R = rng(seed);
    this.x = arena.x0; this.y = arena.y0; this.th = arena.th0;
    this.dEff = [0, 0]; this.prev = null; this.t = 0; this.wob = 0;
    this.swayX = 0; this.swayY = 0;
    const R = this.R;
    if (noise) {
      this.gV = R.uniform(1 - DR_GAIN, 1 + DR_GAIN); this.gW = R.uniform(1 - DR_GAIN, 1 + DR_GAIN);
      this.veer = R.normal(0, DR_VEER);
      this.alpha = DT / Math.max(body.tau * R.uniform(0.7, 2.0), DT);
    } else { this.gV = this.gW = 1; this.veer = 0; this.alpha = DT / body.tau; }
    this.steps = Math.round(DURATION_S[arena.task] / DT);
    this.rec = { x: [this.x], y: [this.y], th: [this.th], drives: [], feats: [], hidden: [] };
  }
  get done() { return this.rec.drives.length >= this.steps; }

  step() {
    const R = this.R, a = this.arena;
    let sx = this.x, sy = this.y, sth = this.th;
    if (this.noise) { sx += this.swayX; sy += this.swayY; sth += this.wob * R.normal(0, DR_YAW_WOBBLE); }
    const s = sense(sx, sy, sth, a.stim, this.prev, this.lesion);
    this.prev = s.state;
    let feats = s.feats;
    if (this.noise) feats = feats.map(f => f + R.normal(0, 0.05));
    const out = act(this.theta, feats, a.task, this.silenced);
    const [dl, dr] = out.drives;
    this.dEff[0] += this.alpha * (dl - this.dEff[0]);
    this.dEff[1] += this.alpha * (dr - this.dEff[1]);
    let [v, u, w] = this.body.velocities(this.dEff[0], this.dEff[1]);
    w = w * this.gW + this.veer * (Math.abs(v) > 0.5 ? 1 : 0);   // veer gated on v before its gain,
    v *= this.gV;                                                // exactly as surrogate.py's tuple assignment
    if (this.noise) { v *= 1 + R.normal(0, 0.08); w += R.normal(0, 0.4); }
    const c = Math.cos(this.th), sn = Math.sin(this.th);
    this.x += (v * c - u * sn) * DT;
    this.y += (v * sn + u * c) * DT;
    this.th += w * DT;
    // the thorax rocks while the legs are active (measured in physics); it is
    // part of what "standing still" is judged on, as in training
    this.wob = clip((Math.abs(this.dEff[0]) + Math.abs(this.dEff[1])) / 2 / STRIDE_DRIVE, 0, 1.3);
    this.swayX = this.wob * R.normal(0, DR_JITTER_MM);
    this.swayY = this.wob * R.normal(0, DR_JITTER_MM);
    this.t += DT;
    const r = this.rec;
    r.x.push(this.x + this.swayX); r.y.push(this.y + this.swayY); r.th.push(this.th);
    r.drives.push(out.drives); r.feats.push(feats); r.hidden.push(out.hidden);
    return out;
  }

  run() { while (!this.done) this.step(); return score(this.arena, this.rec.x, this.rec.y, this.rec.th); }
}

// ------------------------------------------------------------------ scoring
/** tasks.score for one trajectory: pass/fail plus the numbers behind it. */
export function score(arena, xs, ys, ths) {
  const n = xs.length - 1, task = arena.task;
  if (task === "turn") {
    const err = Math.abs(wrap(ths[n] - arena.targetHeading));
    const disp = Math.hypot(xs[n] - xs[0], ys[n] - ys[0]);
    return { success: err < 20 * Math.PI / 180 && disp < TURN_MAX_DISP_MM,
      headingErrorDeg: err * 180 / Math.PI, driftMm: disp };
  }
  const [tx, ty] = targetOf(arena);
  const d = xs.map((x, i) => Math.hypot(x - tx, ys[i] - ty));
  if (task.endsWith("avoid")) {
    const gain = d[n] - d[0], approach = Math.max(d[0] - Math.min(...d), 0);
    return { success: gain > AVOID_GAIN_MM && approach <= AVOID_SLACK_MM, startMm: d[0], endMm: d[n], approachMm: approach };
  }
  const k = Math.round(STOP_WINDOW_S / DT);
  let sp = 0;
  for (let i = n - k; i < n; i++) sp += Math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) / DT;
  sp /= k;
  return { success: d[n] < SUCCESS_RADIUS_MM && sp < STOPPED_MM_S, startMm: d[0], endMm: d[n], finalSpeed: sp };
}

/** 95% Wilson interval for k successes out of n. */
export function wilson(k, n) {
  if (!n) return [0, 1];
  const z = 1.96, p = k / n, den = 1 + z * z / n;
  const c = (p + z * z / (2 * n)) / den, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

// Parity test: does flybrain.js compute exactly what the Python training code does?
//   node check_core.mjs      (after make_reference.py)
//
// Two checks per recorded run:
//  1. Implementation (strict, < 1e-9): before every step the JS fly is put in
//     exactly Python's state (pose, sensory integrator, motor lag), then one step
//     is compared. This isolates the port itself from closed-loop amplification.
//  2. Free run (same verdict, pose < 1e-6 mm): JS runs the whole arena on its
//     own. Rounding differences of ~1e-15 can grow by ~1e6 where a sense is
//     steep (e.g. the light-change input right under the lamp), so a 1e-9 bound
//     on a free run tests floating-point chaos rather than the port; 1e-6 mm
//     (a millionth of a millimetre) and an identical verdict are what matter.
import { readFileSync } from "node:fs";
import { Body, Fly, parseNpy, score } from "../flybrain.js";

const IMPL_TOL = 1e-9, FREE_POSE_TOL_MM = 1e-6;
const here = new URL(".", import.meta.url);
const flylab = new URL("../../", import.meta.url);
const body = new Body(JSON.parse(readFileSync(new URL("../calibration.json", here))));
const ref = JSON.parse(readFileSync(new URL("reference.json", here)));

const maxAbs = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
let worst = { feats: 0, drives: 0, pose: 0, free: 0 }, verdicts = 0, failures = 0;
for (const c of ref.cases) {
  const npy = readFileSync(new URL(`state/brains/${c.task}.npy`, flylab));
  const theta = parseNpy(npy.buffer.slice(npy.byteOffset, npy.byteOffset + npy.byteLength));

  // 1. one step at a time from Python's state
  const step = new Fly(theta, body, structuredClone(c.arena), { noise: false });
  const diff = { feats: 0, drives: 0, pose: 0, free: 0 };
  for (let t = 0; t < c.drives.length; t++) {
    [step.x, step.y, step.th] = c.pose[t];
    step.prev = t ? c.filt[t - 1].slice() : null;
    step.dEff = t ? c.d_eff[t - 1].slice() : [0, 0];
    const out = step.step();
    diff.feats = Math.max(diff.feats, maxAbs(c.feats[t], step.rec.feats[t]));
    diff.drives = Math.max(diff.drives, maxAbs(c.drives[t], out.drives));
    diff.pose = Math.max(diff.pose, maxAbs(c.pose[t + 1], [step.x, step.y, step.th]));
  }

  // 2. free run
  const free = new Fly(theta, body, structuredClone(c.arena), { noise: false });
  for (let t = 0; t < c.drives.length; t++) {
    free.step();
    diff.free = Math.max(diff.free, maxAbs(c.pose[t + 1], [free.x, free.y, free.th]));
  }

  // same trajectory in, same verdict out
  const P = c.pose;
  const s = score(c.arena, P.map(p => p[0]), P.map(p => p[1]), P.map(p => p[2]));
  const same = s.success === c.success;
  verdicts += same;
  const ok = same && diff.feats < IMPL_TOL && diff.drives < IMPL_TOL && diff.pose < IMPL_TOL
    && diff.free < FREE_POSE_TOL_MM;
  failures += !ok;
  for (const k in worst) worst[k] = Math.max(worst[k], diff[k]);
  console.log(`${ok ? "ok  " : "FAIL"} ${c.task.padEnd(13)} ${c.kind.padEnd(6)} ` +
    `step |diff| senses ${diff.feats.toExponential(1)}  drives ${diff.drives.toExponential(1)}  ` +
    `pose ${diff.pose.toExponential(1)}  | free-run pose ${diff.free.toExponential(1)} mm  ` +
    `verdict ${s.success ? "pass" : "fail"}${same ? "" : " (python disagrees)"}`);
}
console.log(`\n${ref.cases.length - failures}/${ref.cases.length} runs match; worst step diffs: ` +
  `senses ${worst.feats.toExponential(1)}, drives ${worst.drives.toExponential(1)}, pose ${worst.pose.toExponential(1)}; ` +
  `worst free-run pose ${worst.free.toExponential(1)} mm; verdicts ${verdicts}/${ref.cases.length}`);
process.exit(failures ? 1 : 0);

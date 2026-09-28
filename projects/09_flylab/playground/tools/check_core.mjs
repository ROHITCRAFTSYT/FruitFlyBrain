// Parity test: does flybrain.js compute exactly what the Python training code does?
//   node check_core.mjs      (after make_reference.py)
import { readFileSync } from "node:fs";
import { Body, Fly, parseNpy, score } from "../flybrain.js";

const here = new URL(".", import.meta.url);
const flylab = new URL("../../", import.meta.url);
const body = new Body(JSON.parse(readFileSync(new URL("../calibration.json", here))));
const ref = JSON.parse(readFileSync(new URL("reference.json", here)));

let worst = { feats: 0, drives: 0, pose: 0 }, verdicts = 0, failures = 0;
for (const c of ref.cases) {
  const npy = readFileSync(new URL(`state/brains/${c.task}.npy`, flylab));
  const theta = parseNpy(npy.buffer.slice(npy.byteOffset, npy.byteOffset + npy.byteLength));
  const fly = new Fly(theta, body, structuredClone(c.arena), { noise: false });
  const diff = { feats: 0, drives: 0, pose: 0 };
  for (let t = 0; t < c.drives.length; t++) {
    const out = fly.step();
    c.feats[t].forEach((v, i) => { diff.feats = Math.max(diff.feats, Math.abs(v - fly.rec.feats[t][i])); });
    c.drives[t].forEach((v, i) => { diff.drives = Math.max(diff.drives, Math.abs(v - out.drives[i])); });
    const p = c.pose[t + 1];
    diff.pose = Math.max(diff.pose, Math.abs(p[0] - fly.x), Math.abs(p[1] - fly.y), Math.abs(p[2] - fly.th));
  }
  // same trajectory in, same verdict out
  const P = c.pose;
  const s = score(c.arena, P.map(p => p[0]), P.map(p => p[1]), P.map(p => p[2]));
  const same = s.success === c.success;
  verdicts += same;
  const ok = same && diff.feats < 1e-9 && diff.drives < 1e-9 && diff.pose < 1e-9;
  failures += !ok;
  for (const k in worst) worst[k] = Math.max(worst[k], diff[k]);
  console.log(`${ok ? "ok  " : "FAIL"} ${c.task.padEnd(13)} ${c.kind.padEnd(6)} ` +
    `max|diff| senses ${diff.feats.toExponential(1)}  drives ${diff.drives.toExponential(1)}  ` +
    `pose ${diff.pose.toExponential(1)}  verdict ${s.success ? "pass" : "fail"}${same ? "" : " (python disagrees)"}`);
}
console.log(`\n${ref.cases.length - failures}/${ref.cases.length} runs match; worst diffs: ` +
  `senses ${worst.feats.toExponential(1)}, drives ${worst.drives.toExponential(1)}, pose ${worst.pose.toExponential(1)}; ` +
  `verdicts ${verdicts}/${ref.cases.length}`);
process.exit(failures ? 1 : 0);

# FlyLab realism backlog

*Last reviewed 2026-11-05. This lists every known gap between FlyLab's
simulation and a real* Drosophila *(or the NeuroMechFly v2 body) that is still
open, ranked by impact relative to cost.*

**Why these are not fixed yet.** Almost every item below changes what the
trained brains sense, how the body moves, or how a task is scored. Any of those
changes voids the published scores of the affected brains, which would then
have to be retrained. The 2026-11-05 review was told not to retrain, so it fixed
only what leaves scores intact: the documentation and comment errors and the
broken parity check, all logged in the
[engineering log](../projects/09_flylab/README.md#making-skills-survive-the-jump-to-physics-engineering-log).
Everything that would cost scores is collected here.

**How to land one.** Change `world.py` / `surrogate.py` / `tasks.py` together,
bump `tasks.TASK_VERSION` for each affected skill, and update the tripwire
tests in `tests/test_flylab_core.py` in the same commit. Those tests fail on
purpose when sensing, the surrogate or the rules change. Retrain on the training
PC, and keep the old champions deployed until the new ones beat them in physics
(the trainer already does this). Port any sensing change to
`playground/flybrain.js` as well, or the parity check goes red. Bundle items
that retrain the same brains, so each brain is retrained once.

Sources: the [2026-09 realism audit](../projects/09_flylab/REALISM_AUDIT.md)
(Fn = its finding n, including its independent fact-check), the measurements in
[`realism_checks.py`](../projects/09_flylab/realism_checks.py), FlyGym 2.1.0
source code, and the papers cited in each item.

## Status of the 2026-09 audit findings

| Audit finding | Status |
|---|---|
| F1 avoid tasks passed by a blind sprint | **fixed** (task rules v2: no approaching the source first) |
| F2 optimistic evaluation statistics | **mostly fixed**: Wilson CIs everywhere, "optimal" needs ≥90% held-out test, daily checks on fresh arenas. Still open: a large one-shot final test (B11) |
| F3 food-seeking rewarded orbiting | **fixed** (v2: within 1.5 mm *and* standing still) |
| F4 surrogate dynamics | open: B1 |
| F5 odor field | open: B2 |
| F6 eye model | open: B3 |
| F7 perfect goal compass | open: B7 |
| F8 fixed-frequency gait, no pivot | open: B5, B6 |
| F9 feedforward brain | open: B8 |
| F10 slow response timing | open: B8 |
| F11 lenient turn task | open, blocked on B6 |
| F12 body scale, arista positions, wobble | **re-verified 2026-11-05** (see "Verified" below) |

## Ranked backlog

| # | Gap | Evidence | Source | Retrains | Cost |
|---|---|---|---|---|---|
| B1 | **Surrogate dynamics.** A 7×7 steady-state table plus a single 0.1 s lag stands in for all transients. Open-loop heading error reaches 11–73° over 3 s, and surrogate success barely predicts physics success on the hard skills | F4 (fact-checked as a "likely major contributor", not a proven cause) | audit F4 | all (warm start) | 1–2 days |
| B2 | **Odor is a static `exp(-r/10 mm)` field.** It gives the same ~1.9% left/right contrast at every distance, below the ~5% asymmetry walking flies are shown to detect, and there is no wind or plume. Physics deployment also senses without the noise used in training | F5, re-derived analytically: contrast = tanh(Δr / 2λ) | Gaudry et al. 2013, *Nature* 493:424, [doi:10.1038/nature11747](https://doi.org/10.1038/nature11747); Álvarez-Salvado et al. 2018, *eLife* 7:e37815, [doi:10.7554/eLife.37815](https://doi.org/10.7554/eLife.37815); NeuroMechFly v2 ships a plume task (Wang-Chen et al. 2024, [doi:10.1038/s41592-024-02497-y](https://doi.org/10.1038/s41592-024-02497-y)) | odor_seek, odor_avoid | ½ day (1/r field + ORN-like compression); days (plume + wind sense) |
| B3 | **Two cosine eyes at ±30°.** They are blind for \|bearing\| ≥ 120°, while real compound eyes are nearly panoramic with a posterior blind spot of roughly 40–50° *(uncertain)*. Brightness `1/(1+(r/30)²)` changes only ~16% across the task range | F6 + fact-check correction (eye axes ≈ ±80°, half-width ≈ 75°, which leaves ~40–50° blind behind) | FlyGym's eye model has 721 ommatidia per eye (`flygym/assets/model/neuromechfly/compound_eye.npz`); Wang-Chen et al. 2024 | light_seek, light_avoid | ½ day |
| B4 | **Vision is computed from the thorax, not the eyes.** The eye bodies sit 0.26 mm ahead of the thorax, so a lamp at 60° and 1 / 2 / 5 mm away is really at 74.5° / 66.9° / 62.6°. The error is largest exactly where `light_seek` has to stop. Light intensity also uses the planar distance and ignores the lamp height, which is not a physical falloff | measured 2026-11-05 (`realism_checks.py`) | NeuroMechFly v2 body (`l_eye` / `r_eye` segments) | light_seek, light_avoid (bundle with B3) | hours |
| B5 | **The gait runs at a fixed 12 Hz.** Speed changes only through stride amplitude (0.33 → 1.36 mm). Real *Drosophila* control walking speed almost exclusively through step frequency | measured yaw peak 11.8 Hz at full stride; FlyGym `cpg_controller.py` `intrinsic_frequency=12.0`; F8 | Wosnitza et al. 2013, *J Exp Biol* 216:480, [doi:10.1242/jeb.078139](https://doi.org/10.1242/jeb.078139); Mendes et al. 2013, *eLife* 2:e00231, [doi:10.7554/eLife.00231](https://doi.org/10.7554/eLife.00231) | all (recalibrate) | 1–3 days CPU; bundle with B1/B6 |
| B6 | **The body can't pivot.** The tightest turn is a ~1.5–1.7 mm radius, and symmetric counter-drives slip sideways faster than they turn. Widening the drive range does **not** help (fact-checked in physics). The turn task therefore allows 6 mm (~2.4 body lengths) of drift | F8 fact-check; `turning_controller.py` only flips the CPG frequency sign | FlyGym 2.1.0 source | all; then tighten `TURN_MAX_DISP_MM` to ≤ 3 mm (F11) | a controller subclass with per-side frequency/phase decoupling, then recalibrate |
| B7 | **The goal compass is perfect.** Exact bearing and distance, with no drift. Real flies keep a central-complex heading estimate that drifts *(drift magnitude for walking flies uncertain)* | F7 | Seelig & Jayaraman 2015, *Nature* 521:186, [doi:10.1038/nature14446](https://doi.org/10.1038/nature14446) | walk_forward, turn, goto | hours |
| B8 | **The brain is feedforward with no memory**, and its 7 task-cue inputs are constant (just biases) inside a per-skill brain. It cannot produce local search after losing an odor. The sense → motion delay is ~0.2 s (20 Hz brain, 0.1 s integrator, 0.1 s motor lag), while flies turn toward odor in less than one stride | F9, F10 | Namiki et al. 2018, *eLife* 7:e34272, [doi:10.7554/eLife.34272](https://doi.org/10.7554/eLife.34272) (hundreds of descending-neuron pairs); Gaudry et al. 2013 | all | days (recurrent layer, ES gets slower) |
| B9 | **The film lamp is drawn too high.** The brain senses the lamp 1.0 mm above the eyes (eyes at 1.17 mm, so ~2.2 mm above ground), but the marker sphere is centred at 4.0 mm. This is rendering only: markers have no collision, so scores are unaffected | measured 2026-11-05; `physics.py` `_add_marker` | NeuroMechFly v2 body | none (re-film only) | ½ hour, but it needs a visual check on the training PC (the sphere would overlap the fly if simply lowered) |
| B10 | **Escape-the-smell is weak when the source is dead ahead (±20°).** It passes 2/10 there: both antennae read the same, so the fly first creeps 0.5–1.25 mm toward the source. This cone is ~11% of random arenas, so daily checks swing between 75% and 100% | engineering log, 2026-10-04 | — | odor_avoid | a measured trial of oversampling frontal sources in surrogate training only |
| B11 | **There is no one-shot final test.** The 32-arena held-out test is re-scored at every new best, so it acts as a stopping criterion. The daily 16 fresh arenas are independent, but small (95% CI 20–30 points wide). Physics is deterministic, so repeating an arena adds nothing, and the only remedy is more arenas | F2 fact-check | audit F2 | none | 1 h code + ~1 h physics per skill (100 arenas, once) |
| B12 | **FlyGym reports an odd `c_head` position.** `c_head` sits 0.09 mm above the ground while the thorax and eyes are at 1.16–1.17 mm. FlyLab never reads `c_head`, so nothing is affected; worth reporting upstream before anyone uses head coordinates | measured 2026-11-05; audit housekeeping | FlyGym 2.1.0 | none | report only |

## Verified on 2026-11-05 (no change needed)

Measured with `realism_checks.py` (FlyGym 2.1.0, MuJoCo 3.9) or read from
FlyGym's own model files:

| Quantity | Model | Reference |
|---|---|---|
| physics timestep | 1e-4 s (10 kHz), Euler, gravity 9810 mm/s² | `mujoco_globals.yaml` |
| leg-controller rate | every 10 steps = 1 kHz (`physics.CTRL_EVERY`) | gait unchanged vs 10 kHz (engineering log) |
| actuated leg DoFs | 42 | NeuroMechFly v2 (Wang-Chen et al. 2024) |
| body mass | 1.00 mg (sum of `rigging.yaml`) | typical fly weight ≈ 10 µN, i.e. ≈ 1 mg (Dickinson & Götz 1996, *J Exp Biol* 199:2085; the conversion is ours) |
| arista offsets | (0.517, ±0.188) mm | `world.py` uses (0.52, ±0.19) |
| full-stride speed | 16.3 mm/s | calibration table 16.2 mm/s |
| gait wobble | 4.38° yaw rms at an 11.8 Hz peak | surrogate uses 4° (`DR_YAW_WOBBLE`) |

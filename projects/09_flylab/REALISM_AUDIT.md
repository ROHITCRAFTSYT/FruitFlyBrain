# FlyLab realism audit

*2026-09-25. This is a measured audit of how closely FlyLab's simulation (surrogate body, senses, brain, tasks and evaluation) matches real* Drosophila melanogaster *and the NeuroMechFly v2 physics body. Every number below was measured for this audit or read from the trainer's own logs (`training/<skill>/status.json`, `log.jsonl`) at the time of writing. The scratch scripts ran at below-normal CPU priority while training continued. The scripts did not change any project file.*

**How to read this.** "Val" means the physics validation arenas that are used for selection. "Test" means the held-out physics arenas that are never used for selection, and those are the honest headline numbers. Intervals are Wilson 95% confidence intervals. Where a biological fact is uncertain, the text marks it *(uncertain)*.

## Ranked findings

Findings are ranked by impact relative to cost. "Retrain" says which brains a fix would force back into training.

| # | Finding (one line) | Impact | Cost | Retrain |
|---|---|---|---|---|
| 1 | **The avoid tasks don't test sensing.** A blind fly that just sprints straight passes **100%** of the physics val and test arenas for both `odor_avoid` and `light_avoid`. | high | low | odor_avoid, light_avoid |
| 2 | **Evaluation statistics are too optimistic.** The same small, fixed val arenas are reused every round (winner's curse). "Optimal" is granted on val, even when test is below 95% (turn 91%, light_avoid 88%). No confidence intervals are reported. | high | low | none |
| 3 | **Food-seeking success rewards orbiting.** 100% of the surrogate successes by the deployed `odor_seek` brain are still moving at the end (median 8.3 mm/s). The body's ~1.5–1.6 mm tightest circle fits inside the 2 mm goal zone. | high | low–med | odor_seek (re-check the others) |
| 4 | **The surrogate's dynamics, not its lookup table, cause the sim-to-sim gap.** The steady-state table is accurate to about 1–12%. On 5 random 3 s drive sequences, however, open-loop heading error reaches 11–73° and position error 1.5–4 mm. Surrogate success barely predicts physics val (r = 0.04–0.37 on the hard skills). | high | medium | all (fine-tune) |
| 5 | **The odor field isn't physical.** `exp(-r/10)` gives a *constant* 1.9% left/right contrast at every distance. Real still-air diffusion (~1/r) gives 15% at 1 mm and 1.3% at 15 mm. Real flies need about 5% (Gaudry 2012). There is also no wind or plume. | medium | medium | odor_seek, odor_avoid |
| 6 | **The eyes are blind to the rear 120°.** The two cosine eyes at ±30° see nothing for bearings beyond ±120°, while the real compound eyes are nearly panoramic. Light intensity changes only 16% across the whole task range. | medium | low–med | light_seek, light_avoid |
| 7 | **The goal compass is perfect.** It gives an exact, noise-free vector and distance to the goal. Real path integration drifts. | medium | low | walk_forward, goto, turn |
| 8 | **The gait has a fixed 12 Hz step frequency.** Speed is set only by stride amplitude (0.33→1.36 mm). The brain's drive range excludes counter-rotation, so the fly can't pivot (minimum radius ≈1.5 mm). | medium | high | all |
| 9 | **The brain is feedforward with no memory.** It can't produce the real OFF-response local search. Its 7 task-cue inputs are constant in per-skill brains. | medium | high | all |
| 10 | **Response timing is ~2× slower than a real fly.** The 0.1 s sensory integrator plus the 0.1 s motor lag at 20 Hz add up to ~0.2 s, while real flies turn toward odor in less than one stride. | low | low | all (small) |
| 11 | **The turn task is lenient.** "Turn in place" allows 6 mm (~2.4 body lengths) of displacement. | low | low | turn |
| 12 | **Validated, no change needed:** body scale, gait-wobble domain randomization, calibration repeatability, and the thorax/arista tracking. | — | — | — |

Suggested order: fix #1, #2 and #3 first. They are cheap, they change what the headline numbers *mean*, and #2 needs no retraining. Then do #4, which gives the biggest gain in transfer. Items #5–#9 make the model more lifelike but each one restarts some brains.

---

## Current headline numbers, with uncertainty (finding 2)

These are from `status.json` at the time of the audit.

| skill | surrogate | physics val (selection set) | **physics test (held out)** | test 95% CI | trainer status |
|---|---:|---:|---:|---|---|
| walk_forward | 100% | 24/24 | **31/32 = 97%** | 84–99% | optimal |
| turn | 99% | 23/24 | **29/32 = 91%** | 76–97% | optimal |
| goto | 100% | 10/12 | **12/16 = 75%** | 51–90% | level 2 |
| odor_seek | 81% | 5/6 | **2/8 = 25%** | 7–59% | level 1 |
| odor_avoid | 100% | 24/24 | **32/32 = 100%** | 89–100% | optimal (see finding 1) |
| light_seek | 100% | 6/6 | **8/8 = 100%** | 68–100% | level 1 passed after 1 round |
| light_avoid | 100% | 23/24 | **28/32 = 88%** | 72–95% | optimal (see finding 1) |

---

## 1. The avoid tasks don't test sensing (high impact, low cost)

**What the model does.** `odor_avoid` places the repellent 2.5–5 mm away and `light_avoid` places the lamp 4–8 mm away. The fly starts with a random heading. Success means ending more than 6 mm farther from the source than at the start, after 4 s.

**Measured.** I drove the **physics** fly with a constant drive (1.2, 1.2), ignoring all senses, for 4 s. It ran 64.7 mm, ending at (62.5, 16.7). I then scored that single trajectory against the trainer's own arena generator (`random_deploy_scenario`, val seeds 10000+i and test seeds 20000+i):

| task | val arenas (24) | test arenas (32) |
|---|---:|---:|
| odor_avoid | blind sprint **100%** | blind sprint **100%** |
| light_avoid | blind sprint **100%** | blind sprint **100%** |

In the surrogate, blind drives of (1.2, 1.2) and (0.6, 0.6) also score 100% on 2000 arenas, with noise on or off. The trained brains' physics errors (29–35 mm gained for odor_avoid, ~40–64 mm for light_avoid) match straight sprints. The 4 light_avoid test failures (gains of −7.2 to +4.2 mm) are cases where the trained brain did *worse* than a blind sprint.

**Reality.** Real flies show odor avoidance, for example turning away from CO₂ or geosmin, and negative phototaxis. What makes the behavior avoidance is that it depends on where the stimulus is. *(The behavior is general knowledge; I measured no specific quantity here.)*

**Proposed change.** Make the blind baseline fail:
- Put the source ahead of the fly (bearing within ±60°, 3–6 mm), so a straight sprint runs into or past it.
- Add a penalty or a failure for coming within about 1.5 mm of the source.
- Optionally bound the arena, so that "get away" means reaching the far side from the source.
- Add a "blind-sprint baseline" row to every skill REPORT, so nobody can mistake a trivial policy for a skill.

**Cost:** a `tasks.py` change plus retraining 2 brains (hours). **Risk:** low. The current "optimal" status of both avoid brains stops meaning anything until they are retrained.

## 2. Statistical honesty of evaluation (high impact, low cost, no retrain)

**What the model does.**
- Physics is deterministic. I re-measured grid points and got identical numbers to 0.01 mm/s.
- The val arenas come from a fixed seed (`val_seed = 10000 + i`), so every round evaluates the same 6/12/24 arenas.
- A new champion is accepted whenever `(val, reward, surrogate)` improves. Across rounds this means taking the maximum of many noisy candidates on the same arenas.
- The level-up val sets are nested: the 12 level-2 arenas contain the 6 level-1 arenas that were already selected on.
- "Optimal" requires val ≥ 95% on 24 arenas. It does not look at test.

**Measured.**

| skill | rounds on the same val set | mean (val − test) at new bests | best: val vs test |
|---|---:|---:|---|
| odor_seek | 47 | **+0.24** | 5/6 (83%) vs 2/8 (25%) |
| goto | 29 | +0.10 | 10/12 vs 12/16 |
| light_avoid | 10 | +0.08 | 23/24 vs 28/32 |
| turn | 3 | +0.06 | 23/24 vs 29/32 |
| walk_forward | 25 | +0.01 | 24/24 vs 31/32 |
| odor_avoid | 42 | −0.16 | 24/24 vs 32/32 |

How easy is a level to pass by luck? A brain whose true success rate is 80% passes 6/6 with probability 0.26 per round, and **0.998** within 20 rounds. At p = 0.7 the chance within 20 rounds is 0.92. Level 3 (≥ 23/24) passes with probability 0.29 at a true rate of 90%, and 0.49 at 93%, per attempt.

Two brains are labelled **optimal** although their held-out test is below the 95% bar: turn at 91% (CI 76–97%) and light_avoid at 88% (CI 72–95%). light_seek's 8/8 has a CI lower bound of only 68%.

There is also some human-in-the-loop leakage. Engineering fixes, such as the odor_seek reflex, were chosen after looking at test results. That makes the test set slightly less "held out" than it looks. *(This is a qualitative point.)*

**Proposed changes** (all in `train_brains.py` / reporting, no retraining):
1. Print Wilson 95% CIs next to every val and test percentage in TRAINING.md and the REPORTs.
2. Define **optimal** by the held-out test: for example, test ≥ 95% on 32 arenas, or a test CI lower bound ≥ 85%.
3. Draw fresh val arenas each round (seed = round), or use a rotating pool, so selection can't overfit a fixed handful of arenas. Alternatively, re-score a champion on new arenas before accepting it.
4. Keep a final, larger test set (e.g. 100 arenas per skill) that is scored only once, when a skill is declared done.

**Cost:** about 1 hour of code. **Risk:** some skills lose their "optimal" label, which is the honest outcome.

## 3. Food-seeking success rewards orbiting (high impact, low–medium cost)

**What the model does.** In `tasks.score`, success for go-to tasks is `final distance < 2.0 mm`, whatever the speed. Reward adds dwell time inside 2 mm. `world.REACH_RADIUS_MM = 1.5` is defined but never used.

**Measured.** I ran the deployed brains in the surrogate on 256 random arenas with training noise on, looking at the last 1 s of each episode:

| brain | success | median final distance | successes still moving > 1 mm/s | median speed of successes |
|---|---:|---:|---:|---:|
| walk_forward | 100% | 0.06 mm | 1% | 0.56 mm/s |
| goto | 100% | 0.06 mm | 0% | 0.54 mm/s |
| **odor_seek** | 73% | **1.73 mm** | **100%** | **8.29 mm/s** |
| light_seek | 100% | 0.09 mm | 84% (creeping; stays within 0.24 mm) | 1.50 mm/s |

With noise off, odor_seek scores 83% and still has 100% of its successes moving. The calibration table shows why orbiting works. At the brain's extreme drives (−0.5, 1.2) the body turns at 313°/s while moving at 8.7 mm/s, which is a 1.59 mm radius. Every high-rate turn in the grid has a radius of 1.47–1.62 mm, so the tightest circle fits inside the 2 mm success zone.

**Reality.** A fly that finds food stops, extends its proboscis and feeds. Local search around a lost odor is a separate behavior (Álvarez-Salvado et al. 2018, [doi:10.7554/eLife.37815](https://doi.org/10.7554/eLife.37815)).

**Proposed change.** Success for odor_seek and light_seek should be "within 1.5 mm (REACH_RADIUS_MM) **and** speed < 1 mm/s over the last 0.5 s". Dwell reward should only count while slow. Reports should show a "stopped at goal" rate next to success.

**Cost:** a `tasks.py` change. odor_seek must retrain, which the existing reflex seed should make feasible because the hand-written reflex stops at a median of 0.47 mm (README). walk_forward and goto already stop, so their numbers should barely move. **Risk:** light_seek may need 1–3 more rounds.

## 4. The sim-to-sim gap comes from the surrogate's dynamics (high impact, medium cost)

**What the model does.** The surrogate looks up steady-state forward, lateral and turning velocity in a 7×7 table of left/right drives, measured over 0.9 s after a 0.4 s settle. The table is interpolated bilinearly. A single first-order lag, τ = 0.1 s, measured from the forward-speed step response, stands in for all transients. Domain randomization adds ±20% gains, a 0.25 rad/s veer, and 0.4 rad/s of turn noise.

**Measured: steady state.** Physics against the surrogate lookup, holding each drive constant:

| drive L, R | physics v / lat / ω | surrogate v / lat / ω |
|---|---|---|
| 1.2, 1.2 (grid) | 16.17 / −0.37 / −8.1°/s | identical |
| −0.5, 1.2 (grid) | 8.67 / +2.86 / +313.2°/s | identical |
| 0.75, 0.75 | 10.83 / −0.06 / −6.7 | 10.67 / −0.06 / −4.5 |
| 0.45, 1.05 | 10.52 / +1.61 / +114.4 | 10.43 / +1.53 / +112.6 |
| 1.05, 0.20 | 7.96 / −2.34 / −178.3 | 7.92 / −2.26 / −174.8 |
| −0.25, 1.05 | 8.52 / +2.22 / +264.2 | **7.46** / **+3.04** / 251.9 (12% speed error) |
| 0.20, 0.20 | 2.38 / 0.00 / −4.7 | 2.36 / 0.00 / −2.9 |

The table is good: under 2% error off-grid in most cells, and up to 12% where a negative drive is interpolated.

**Measured: dynamics.** I drove physics and the surrogate open-loop with the same random drive sequences for 3 s. Each drive was held for 0.25–0.5 s, or for 1 s in the last row.

| sequence | heading error at 0.5 / 1 / 3 s | position error at 3 s | physics path length |
|---|---|---:|---:|
| seed 1 | 28° / 32° / **73°** | 3.8 mm | 15.9 mm |
| seed 2 | 9° / 1° / 11° | 1.5 mm | 15.2 mm |
| seed 3 | 1° / 5° / 15° | 2.6 mm | 16.0 mm |
| seed 4 | 16° / 20° / 14° | 1.6 mm | 29.6 mm |
| seed 5 (1 s holds) | 13° / 15° / 14° | 1.9 mm | 17.9 mm |

The RMS turning-rate error per 50 ms step is 53–126°/s. Part of that is the 12 Hz gait yaw oscillation. The rest is error in the transients.

**Consequence in training data** (`log.jsonl`, all rounds):

| skill | mean surrogate | mean physics val | mean gap | corr(surrogate, physics val) across rounds |
|---|---:|---:|---:|---:|
| walk_forward | 0.73 | 0.49 | +0.23 | 0.35 |
| goto | 0.82 | 0.63 | +0.19 | 0.37 |
| odor_seek | 0.83 | 0.45 | **+0.38** | **0.04** |
| odor_avoid | 0.90 | 0.67 | +0.22 | 1.00 (few distinct values) |
| light_avoid | 1.00 | 0.76 | +0.24 | −0.14 |

For the hard skills, surrogate success is nearly uninformative about physics success. Progress therefore depends on physics selection over a few arenas, and that selection is exactly where the winner's curse in finding 2 bites.

**Proposed change.**
1. Log about 5–10 minutes of physics under random drive sequences. This is cheap: each 3 s sequence took ~30 s wall-clock at low priority.
2. Fit a *dynamic* surrogate from those logs. Options are separate lags for v and ω, dependence on drive history and gait phase, or a small residual regressor for Δpose over 50 ms. Keep the table as the prior.
3. Validate on held-out sequences, targeting < 15° heading error at 3 s, and report that fit error in the README.
4. Scale the domain randomization (veer, turn noise) to the *measured* residuals instead of guessing.

**Cost:** 1–2 days. All brains need fine-tuning, but they can warm-start from their current weights. **Risk:** medium. A better surrogate can briefly lower surrogate scores. Keep the old champions deployed until the new ones win on physics.

## 5. Olfaction: odor field and antennae (medium impact, medium cost)

**What the model does.**
- Intensity is `exp(-r / 10 mm)`, a static field with no wind and no intermittency.
- The antennae sit at the *measured* arista positions (0.52 mm forward, ±0.19 mm). I re-measured those offsets in physics and they match.
- The feature is the contrast `(L−R)/(L+R)` × 20, plus Gaussian noise with sd 0.05 in training.

**Measured.** Raw bilateral contrast with the source at 90° to one side:

| distance | exp(−r/10) (model) | 1/r (3D still-air diffusion) | model feature (noise sd 0.05) |
|---:|---:|---:|---:|
| 1 mm | 1.68% | 14.9% | 0.34 |
| 2 mm | 1.84% | 8.9% | 0.37 |
| 5 mm | 1.89% | 3.8% | 0.38 |
| 11 mm | 1.90% | 1.7% | 0.38 |
| 15 mm | 1.90% | 1.3% | 0.38 |

The exponential field has a constant relative gradient (1/λ), so the fly gets the *same* steering signal at 15 mm as at 2 mm. With the source straight ahead, a 4° gait yaw wobble moves the feature by only ±0.03.

**Reality.**
- Gaudry et al. 2012 ([doi:10.1038/nature11747](https://doi.org/10.1038/nature11747)) found that a walking fly can detect about a **5%** left/right asymmetry in ORN input and turns within less than one stride. The model's constant 1.9% concentration contrast sits below that. Amplified and with low noise, it becomes a reliable steering signal even at 15 mm, which is probably easier than reality. *(The mapping from concentration contrast to ORN-input asymmetry is uncertain.)*
- Real walking flies mostly navigate odor by upwind runs during odor, which need wind sensing through the antennal mechanoreceptors, and by local search after the odor is lost (Álvarez-Salvado et al. 2018, [doi:10.7554/eLife.37815](https://doi.org/10.7554/eLife.37815)).
- NeuroMechFly v2 already ships a simulated turbulent odor-plume task (Wang-Chen et al. 2024, [doi:10.1038/s41592-024-02497-y](https://doi.org/10.1038/s41592-024-02497-y)).

**Proposed change, staged:**
- (a) A cheap stage: use a 1/r (or 3D-diffusion) field with a log-compressed ORN-like response, and calibrate the contrast noise so that about 5% asymmetry is the detection threshold. Retrain the two odor brains.
- (b) A realistic stage: add a wind vector and a wind-direction sense, and train in FlyGym's plume.

Expect (a) to make far-field odor_seek *harder*, which is realistic and requires searching or casting. That strengthens the case for memory (finding 9). **Cost:** (a) about half a day plus retraining; (b) several days. **Risk:** the odor_seek numbers will drop at first.

## 6. Vision: eye model (medium impact, low–medium cost)

**What the model does.** There are two eyes with cosine tuning `max(0, cos(bearing ∓ 30°))`. Brightness is `1/(1+(r/30 mm)²)`. There is no pattern vision.

**Measured.**
- Both eyes return exactly 0 for **122° of bearings (|bearing| ≥ 120°)**. A light behind the fly is invisible.
- Across the task ranges, intensity barely changes: 0.934 at 8 mm and 0.779 at 16 mm for light_seek, 0.98–0.93 for light_avoid. Distance information is weak.

Because the rear sector is blind, "hide from light" can end in a state where the fly simply can't see the lamp any more.

**Reality.** Each compound eye has about 750–800 ommatidia. Together the two eyes give a nearly panoramic field of view, with a small rear blind zone and some frontal binocular overlap *(the exact angles are uncertain; roughly 300°+ total coverage is commonly cited)*. NeuroMechFly v2 includes a compound-eye renderer with 721 ommatidia per eye (Wang-Chen et al. 2024, [doi:10.1038/s41592-024-02497-y](https://doi.org/10.1038/s41592-024-02497-y)).

**Proposed change.** Cheap version: give each eye a broad lateral hemifield, with the eye axis at ±90° and a raised-cosine half-width of about 100°, so the rear blind zone shrinks to roughly 20–30°. Add a looming or angular-size cue for distance. Full version: read summed ommatidial intensities from FlyGym's eye renderer (slow on 2 cores). **Cost:** the cheap version is a `world.py` change plus retraining the 2 light brains (light_seek mastered level 1 in a single round). **Risk:** low.

## 7. Goal compass is perfect (medium impact, low cost)

**What the model does.** `walk_forward`, `goto` and `turn` receive the exact sin and cos of the goal bearing and `tanh(distance/10)`, with no drift.

**Reality.** Flies keep a heading estimate in the central complex (Seelig & Jayaraman 2015). They can path-integrate, but the estimate drifts and has noise *(the magnitude for walking* Drosophila *is uncertain)*.

**Proposed change.** Add heading-estimate noise and a drift that grows with distance walked (for example 5–10° per 10 mm), plus multiplicative noise on distance. **Cost:** a small change to `world.py`/`surrogate.py` and the physics sensing, then retraining 3 brains. **Risk:** low. It will probably turn a 0.06 mm median stop into a realistic ~0.5–1 mm one.

## 8. Gait, speed range, turning radius and pivoting (medium impact, high cost)

**Measured on the physics body:**

| quantity | FlyLab / NeuroMechFly | real *Drosophila* |
|---|---|---|
| body length | ~2.4 mm between geom centers, from abdomen tip to arista (CT-based model) | ~2.5 mm ✓ |
| top speed (drive 1.2, 1.2) | 16.2 mm/s (~6.5 body lengths/s) | free walking spans a few to ~30+ mm/s depending on assay (Mendes 2013 [doi:10.7554/eLife.00231](https://doi.org/10.7554/eLife.00231); DeAngelis 2019 [doi:10.7554/eLife.46409](https://doi.org/10.7554/eLife.46409)) *(exact range uncertain)*. The model covers the slow-to-moderate half. |
| step frequency | **fixed 12.0 Hz** at 4 and 16 mm/s (FFT of yaw) | rises with speed. Real flies change speed mainly through stance duration/frequency (DeAngelis 2019; Mendes 2013). |
| stride length | 0.33 mm at 4 mm/s → 1.36 mm at 16 mm/s | changes less than frequency *(uncertain)* |
| max turning rate | ~310–320°/s in the brain's range | walking body saccades reach several hundred °/s *(uncertain)* ✓ |
| tightest turning radius | 1.47–1.62 mm at every high-rate drive pair in the brain's range | flies can turn nearly on the spot *(uncertain)* |
| counter-drives outside the brain's range | (−0.8, 0.8): 396°/s, radius 0.95 mm, but 7.4 mm/s sideways slip. (−1.2, 1.2): 355°/s, 9.4 mm/s slip | — |
| gait wobble at full speed | lateral 0.18 mm rms, yaw 4.4° rms | (matches the domain randomization of 0.15 mm and 4°) ✓ |

FlyGym's `HybridTurningController` sets CPG *amplitude* to |drive| and flips the CPG frequency sign for a negative drive. The tripod coupling across sides then fights opposite-direction stepping, which produces the sideways slip instead of a clean pivot.

**Proposed change.**
- (a) Widen the brain's drive range to about [−1.0, 1.2] and re-calibrate. This gives the brains a ~1 mm turning radius, which removes the orbit-inside-the-goal loophole physically, not just in the reward.
- (b) Subclass the turning controller (a new file; FlyGym itself stays untouched) so that |drive| also scales CPG frequency, for example 6–15 Hz. That makes slow walking look real.

**Cost:** recalibration plus retraining all brains, 1–3 days of CPU. **Risk:** high, because it changes the body every brain learned. Do it together with finding 4 so the brains are retrained only once.

## 9. The brain itself (medium impact, high cost)

**What the model does.** A 17→32(tanh)→2 MLP that outputs left and right "descending drives" in [−0.5, 1.2], with a motor gate. Seven of the 17 inputs are a one-hot task cue that is **constant** inside a per-skill brain, so they act only as extra biases. The network has no recurrent state. The only memory is the sensory integrator and the "dt" features.

**Reality.**
- The two-drive interface is the same abstraction NeuroMechFly v2 uses for its hybrid turning controller (Wang-Chen 2024), so it is a fair level of description.
- Real flies send commands through hundreds of pairs of descending neurons (Namiki et al. 2018, [doi:10.7554/eLife.34272](https://doi.org/10.7554/eLife.34272)). A few identified DNs control steering, forward walking and backward walking: DNa01/DNa02 for steering, P9/DNp09 for forward walking, and the moonwalker DN (MDN) for backward walking *(attributions from memory; cite the primary papers before publishing)*.
- Real odor navigation needs memory, for example the search at odor offset (Álvarez-Salvado 2018).

**Proposed change.** Drop the constant cue inputs. Add a small recurrent layer (leaky units or a GRU with 8–16 units) so the brains can cast, search and time their stops. Optionally add a third output, a "stop/halt" DN-like command, alongside the two steering drives. The longer-term step is to constrain the hidden layer with central-complex connectivity, as the README already notes. **Cost:** changes to brain.py and trainer, then retraining all brains. **Risk:** ES with more parameters is slower on 2 cores.

## 10. Timing (low impact, low cost)

**Measured.** The brain runs at 20 Hz. The sensory integrator has τ = 0.098 s and the motor/CPG lag has τ = 0.10 s (from `state/calibration.npz`). The effective sensory-to-motion response is about 0.2 s. The model has no pure transport delay.

**Reality.** Gaudry et al. 2012 found walking flies turn toward an odor asymmetry "in less time than it requires the fly to complete a stride", which is under ~80–100 ms at 10–12 Hz stepping. Early visual processing has latencies of tens of ms *(uncertain)*.

**Proposed change.** Reduce the integrator to τ ≈ 0.05 s once the surrogate's dynamics are better (finding 4), or run the brain at 40 Hz. Model an explicit 20–30 ms delay. **Cost:** small, but it retrains all brains. Bundle it with finding 4 or 8.

## 11. The turn task is lenient (low impact, low cost)

**What the model does.** Success means heading error under 20° and displacement under 6 mm, about 2.4 body lengths. The README records the reason: the gait "can't pivot perfectly".

**Proposed change.** Once finding 8(a) allows tighter turns, tighten the limit to ≤ 3 mm (about one body length) and log the displacement in `physics_errors` next to the angle.

## 12. Checked and found realistic (no change)

- **Scale.** NeuroMechFly v2 comes from a micro-CT scan of a real female fly. The geom centers span ~2.4 mm from the abdomen tip to the aristae. The thorax body sits 1.16 mm above the ground.
- **Arista offsets.** I re-measured (1.07−0.55, ±0.19) = 0.52 mm forward and ±0.19 mm. `world.py` matches.
- **Gait-wobble domain randomization.** The measured 0.18 mm lateral and 4.4° yaw rms at full stride agree with `DR_JITTER_MM = 0.15` and `DR_YAW_WOBBLE = 4°`.
- **Calibration is reproducible.** Physics is deterministic, so re-measured grid cells matched to 0.01 mm/s. The flip side is that the physics fly has no trial-to-trial variability, so repeating an arena adds no information. All the uncertainty in finding 2 comes from how few arenas there are.

### Minor housekeeping (no behavioral impact)
- `world.REACH_RADIUS_MM` is defined but unused. Finding 3 proposes using it.
- The README says the calibration uses "36 combinations", but the grid is 7×7 = 49, which is what calibrate.py prints.
- The FlyGym API returned an implausible position for the `c_head` segment: 2.15 mm *behind* the thorax, at z = 0.09. The thorax and arista positions agree with MuJoCo's `xpos`, so FlyLab is unaffected. Check this before anyone uses head coordinates.

## Method

- **Logs.** `training/*/status.json` and `log.jsonl` were read-only. The CIs are Wilson intervals, and the luck probabilities are binomial.
- **Surrogate.** Deployed brains from `state/brains/*.npy` were loaded read-only. I ran 256 arenas per skill with training noise (seeds 7/8), and 2000 arenas for the blind baselines.
- **Physics.** FlyGym 2.1 + MuJoCo, using `physics.PhysicsFly` exactly as the trainer does, at below-normal priority. The runs were 8 steady-state checks, 3 pivot tests, 2 stride/wobble recordings of 1 s at 1 kHz, 1 blind 4 s sprint scored against the trainer's val/test arena generators, and 5 open-loop 3 s drive sequences. That is about 45 s of simulated time and ~10 minutes of wall-clock time.
- The scratch scripts live outside the project, in the session scratchpad. No training file, state file or process was touched.

## Independent fact-check

A second agent tried to refute every finding: it checked the facts against sources and FlyGym code and re-ran measurements. Its verdicts and corrections take precedence over the text above.

| finding | verdict | notes | correction |
|---|---|---|---|
| F1 | **partly** | The claim holds. I re-ran the 4 s constant (1.2, 1.2) physics sprint at below-normal priority and got the same endpoint to the hundredth of a mm, (62.51, 16.67), which also confirms physics is deterministic. Scored against random_deploy_scenario with val seeds 10000+i and test seeds 20000+i, it passes 24/24 val and 32/32 test for both odor_avoid and light_avoid. A noise-free surrogate sprint, ending at (60.4, -18.4), also passes 100%. The code matches: odor source 2.5-5 mm, light source 4-8 mm, random heading, success is gain > 6 mm (tasks.score). The proposed fix does not work, though. In the surrogate I tested blind policies on the proposed arenas (source 3-6 mm, bearing within +/-60 deg, fail if closer than 1.5 mm, 4000 arenas). A straight sprint still passes 67%. A blind U-turn followed by a sprint passes 92%. A blind backward walk, drive (-0.5, -0.5), which is -3.8 mm/s in the calibration table, covers about 15 mm in 4 s and passes 100%. Putting the source always ahead just swaps one blind solution for another. | The finding is correct, but the fix must beat every blind policy, not only the forward sprint. Keep bearings uniform over 360 deg, or use a mix of ahead and behind, so no single open-loop action works. Add the near-approach failure and/or a bounded arena. Make 'every blind baseline (sprint, backward, U-turn-then-sprint, constant turn) scores at or near chance' an acceptance test for the new task definition, not just a REPORT row. |
| F2 | **confirmed** | Code and logs check out. val_seed is 10000+i and fixed, so the same arenas are used every round. I checked that the level sets are nested: the first 6 of the 12 goto val arenas are identical to the level-1 set. optimal = level 3 and level_met, which tests only physics_val >= 0.95 and surrogate >= 0.95, never test. status.json has turn at val 23/24 and test 29/32, and light_avoid at val 23/24 and test 28/32, both at level 3 and so 'optimal'. My recomputed Wilson intervals: 2/8 gives 7.1-59.1%, 29/32 gives 75.8-96.8%, 28/32 gives 71.9-95.0%, 8/8 gives 67.6-100%. Binomials: 0.8^6 = 0.262 and 1-(1-0.262)^20 = 0.998; at p = 0.7 it is 0.918; P(>= 23/24) is 0.29 at p = 0.9 and 0.49 at p = 0.93. Mean val-minus-test gap at new bests from log.jsonl: odor_seek +0.244, goto +0.101, light_avoid +0.083, turn +0.056, walk_forward +0.005, odor_avoid -0.156. grep finds no CI or Wilson in TRAINING.md, the REPORTs or the README. One caveat on the proposal: if 'optimal' is gated on the 32-arena test that is re-scored at every new best, the test becomes a selection set. The proposed one-shot final test (about 100 arenas) is needed to keep a truly held-out number. |  |
| F3 | **confirmed** | tasks.score returns success = df < SUCCESS_RADIUS_MM (2.0) with no speed term. grep shows REACH_RADIUS_MM appears only at its definition in world.py. I re-ran the current deployed brains in the surrogate on 256 arenas. odor_seek: success 76% with noise and 70% without, median final distance 1.68-1.82 mm, 100% of successes moving faster than 1 mm/s in the last second (median 8.7-9.1 mm/s). Under the proposed strict criterion (within 1.5 mm and under 1 mm/s over the last 0.5 s) odor_seek scores 0%. walk_forward and goto stop (median 0.03-0.05 mm, 0-1% moving) and score 99-100% under the strict criterion. light_seek creeps but still scores 99-100% strict, so the proposal would not break it. Calibration at (-0.5, 1.2) is v 8.67 mm/s at 313 deg/s, giving R_fwd 1.59 mm. High-rate cells range 1.47-1.62 mm using forward speed only, or 1.6-1.9 mm using total speed including slip, which is still inside 2 mm. Small drift: the brain changed since the audit, so the audit's 73% is now 76%. Orbiting was shown in the surrogate only, not in physics. |  |
| F4 | **partly** | Confirmed parts. calibrate() fits a single tau from the forward-speed step response only, quantised to 25 ms (tau = 0.10 in calibration.npz), and one lag is used for all channels. From log.jsonl, the mean surrogate-minus-physics-val gap is +0.19 to +0.39, and the correlation across rounds is 0.35 for walk_forward, 0.37 for goto and 0.06 for odor_seek (the audit said 0.04; one more round has run since). My sprint re-run independently supports a transient problem. Under a constant (1.2, 1.2) drive, physics heading ends at +15 deg, while the steady-state table (-8.1 deg/s) makes the surrogate drift to about -32 deg, so the endpoints differ by about 35 mm laterally. Not reproduced: the five random-sequence heading and position errors and the 53-126 deg/s RMS. Not established: the causal link. The across-round correlation is a weak test, because physics val uses only 6-24 deterministic arenas and surrogate success often saturates at 100%. Physics sensing also differs from the surrogate: it has no feature noise, and its gait wobble is real rather than randomised. The gap is not proven to come mainly from transients, although that is plausible. | Present the transient mismatch as a likely major contributor, not a demonstrated cause. Before claiming causality, validate the proposed dynamic surrogate by checking that closed-loop brain success in the new surrogate predicts physics success better, not only by open-loop heading error. |
| F5 | **confirmed** | Re-derived with world.antenna_positions and the source 90 deg to the side. exp(-r/10) contrast is 1.68% at 1 mm, 1.84% at 2 mm, 1.89% at 5 mm and 1.90% at 11-15 mm, and the x20 feature is 0.34-0.38. A 1/r field gives 14.86%, 8.89%, 3.76%, 1.72% and 1.27%. This is analytically true: the contrast is tanh(delta_r / 2 lambda), which is constant in the far field. The feature against noise sd 0.05 is correct (world.sense, surrogate._run), and physics sensing has no noise at all. Web check of Gaudry et al. (Nature 493:424, doi:10.1038/nature11747): a walking fly can detect a 5% asymmetry in total ORN input and turns in less time than one stride. The Alvarez-Salvado 2018 eLife DOI and the NeuroMechFly v2 plume task (Wang-Chen 2024) are real references. One nuance: the model's contrast is below 5% at every distance, so by the Gaudry threshold the model makes the task easier at all distances, not only in the far field. The link from concentration contrast to ORN input is uncertain, as the audit notes. |  |
| F6 | **partly** | Confirmed: max(0, cos(b -/+ 30 deg)) is zero for both eyes when \|b\| >= 120 deg, and on a 1-deg grid that is 122 of 361 samples. The actual blind sector is 120 deg; 122 is the sample count. Intensities are 0.983 at 4 mm, 0.934 at 8 mm and 0.779 at 16 mm, and FlyGym's compound_eye.npz and vision.yaml give 721 ommatidia per eye. The proposal is off in two ways. First, eyes at +/-90 deg with a raised-cosine half-width of 100 deg overlap behind the fly, so the blind zone is 0 deg, not 20-30 deg; I computed a blind count of 0 and a minimum rear sensitivity of 0.05. Second, published measurements put the real posterior blind spot at about 40-50 deg (roughly 40 deg in older optical data, about 50 deg in the recent eye-structure maps, Nature 2025, s41586-025-09276-5), with a frontal overlap under 20 deg. So the proposed 20-30 deg, or 0 deg with those parameters, over-corrects. | The blind sector is 120 deg, not 122. Real flies have a rear blind spot of about 40-50 deg. Choose eye axes and half-widths that leave about 40-50 deg behind, for example axes at +/-80 deg with half-width about 75 deg. Eyes at +/-90 deg with half-width 100 deg leave no blind zone. |
| F7 | **confirmed** | In world.sense the goal features (sin, cos, tanh(dist/10)) have no noise or drift. The surrogate adds only the generic 0.05 feature noise. physics.deploy calls sense() with no noise at all. My surrogate re-run gives median final distances of 0.03-0.05 mm for walk_forward and goto, consistent with the audit's 0.06 mm. The biology is stated with appropriate hedging: the central complex heading estimate is real, and the drift magnitude is uncertain. The proposed numbers are only examples and I did not check them. |  |
| F8 | **partly** | Confirmed parts. The FlyGym HybridTurningController (flygym_demo/complex_terrain/turning_controller.py) sets CPG amplitude to \|drive\|, and a negative drive only flips the sign of the base intrinsic frequency. My physics yaw FFT peaks at 12.0 Hz at 16 mm/s and 12.2 Hz at 4.0 mm/s, which gives stride lengths of about 1.35 and 0.33 mm. brain.py limits drives to [-0.5, 1.2]. The high-rate calibration cells have forward-speed radius 1.47-1.62 mm. The Mendes 2013 eLife finding that speed changes mainly through stance duration and frequency checks out. Wrong or unsupported parts. The range does include one-sided counter-stepping down to -0.5, and brain.py itself says negative drive means spot turns; what it excludes is full symmetric counter-rotation. More importantly, proposal (a) failed my physics test, run at low priority for 1.3 s per drive. At (-1.0, 1.2): v 6.16 mm/s, lateral 4.63 mm/s, 258 deg/s, so the path radius is about sqrt(6.16^2 + 4.63^2) / 4.50, roughly 1.7 mm, no tighter than the current (-0.5, 1.2). At (-0.8, 0.8): 400 deg/s, but lateral slip of 6.7 mm/s is larger than forward speed of 4.6 mm/s, so the path radius is about 1.2 mm (a circle fit gives about 1.5 mm). The audit's 0.95 mm used forward speed only and contradicts its own 7.4 mm/s slip figure. Widening the range to [-1.0, 1.2] will not give a clean turning radius of about 1 mm with this controller. | Widening the drive range alone does not reach a 1 mm radius. Asymmetric counter-drives such as (-1.0, 1.2) give about 1.7 mm, and symmetric (-0.8, 0.8) gives about 1.2-1.5 mm with slip larger than forward speed. Real pivoting needs a controller change, such as frequency and phase decoupling between sides in the new controller subclass. Compute radius from total planar speed, not forward speed. Reword 'excludes counter-rotation' to 'allows only partial one-sided counter-stepping (down to -0.5)'. |
| F9 | **confirmed** | brain.py is a feedforward 17 -> 32 (tanh) -> 2 network, with N_IN = N_SENSES 10 + N_TASKS 7. task_cue is a constant one-hot inside a per-skill brain, so it acts only as a bias. There are no recurrent state variables; memory exists only through the sensory integrator (world.SENSE_BETA) and the dt features. The two-drive interface matches the FlyGym HybridTurningController's 2-element descending_signal. Namiki et al. 2018 (eLife 7:e34272) exists and describes hundreds of DN pairs (they estimate about 350-500 pairs and characterised about half), so 'hundreds' is correct. I did not test whether the recurrent-units proposal would enable search at odor offset; it is a design suggestion. |  |
| F10 | **confirmed** | SENSE_BETA = 0.4 per 50 ms step gives tau = -0.05/ln(0.6) = 0.098 s. calibration.npz has tau = 0.10 s. The brain runs at 20 Hz (DT = 0.05). Two cascaded first-order lags have a total mean delay of about 0.2 s, plus up to 50 ms of control-step quantisation. Web check of Gaudry et al.: walking flies turn toward odor in less time than one stride, and at about 10-12 Hz a stride is 80-100 ms. The low-impact rating and the plan to bundle this with F4/F8 are sensible. |  |
| F11 | **confirmed** | tasks.TURN_MAX_DISP_MM = 6.0, and turn success is err < 20 deg and disp < 6 mm. train_brains._error_of returns only the angle in degrees for turn, so displacement is never logged. 6 mm is about 2.4 body lengths if a body is about 2.5 mm. The proposal depends on tighter turns, which F8(a) as proposed does not deliver (see F8), so tightening to 3 mm or less may need the controller change first. |  |
| F12 | **partly** | Verified parts. world.py has ANTENNA_FORWARD = 0.52 and ANTENNA_HALF_SPAN = 0.19. REACH_RADIUS_MM is unused (grep). README line 68 says '36 combinations', while calibrate's default grid has 7 values (49 cells), and calibration.npz has a 7-value grid. Physics is deterministic: my re-run of the 4 s sprint reproduced the audit's endpoint (62.51, 16.67) exactly. The DR constants 0.15 mm and 4 deg are as stated. Not re-measured: the 0.18 mm / 4.4 deg wobble, the 2.4 mm geom span, the arista positions from MuJoCo xpos, and the c_head anomaly. FlyLab code never references c_head (grep), so that item is harmless either way. One small audit wording slip: the blind sprint 'ran 64.7 mm' is the net displacement; the path length at 1 kHz sampling including wobble is about 96 mm. | The housekeeping items are confirmed. The wobble, scale and c_head sub-claims are unverified by this check, not refuted. |

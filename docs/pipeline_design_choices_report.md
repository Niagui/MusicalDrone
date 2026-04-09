# Pipeline Design Choices and Heuristics Report

## Purpose

This document summarizes plausible justifications for the design choices and heuristics currently used in the drone choreography pipeline. The goal is not to claim that every heuristic is already empirically optimal. Instead, it gives defensible explanations for why each choice is reasonable in a capstone-style system that must balance musical expressiveness, interpretability, robustness, and swarm safety.

Where helpful, this report distinguishes between:

- core pipeline choices that are currently wired into the main run path
- optional or newer extensions that are implemented but not always required
- safety heuristics whose purpose is feasibility rather than aesthetics

## High-Level Pipeline View

At a high level, the project turns audio into swarm motion through the following stages:

1. Extract timing structure from the audio.
2. Represent each time segment with affective labels.
3. Map labels into a smaller set of reusable emotional anchors.
4. Convert anchor weights into boid parameters.
5. Run the swarm inside a bounded and safety-constrained controller.

There are also two important optional extensions:

- an LLM-based label variation step that enriches the emotion vocabulary
- an LLM-based phrase planner that adds phrase-level attractor motion on top of the base swarm behavior

## Framing for the Paper

The strongest overall justification is:

> Because there is no ground-truth dataset that directly maps music segments to safe and expressive drone-swarm trajectories, the system uses a hybrid pipeline. Learned models are used where semantic understanding is valuable, while hand-designed heuristics are used where interpretability, musical timing, and safety constraints matter most.

This framing helps position the heuristics as deliberate engineering choices rather than arbitrary guesses.

## 1. Audio Timing and Segmentation Choices

### 1.1 Resampling audio to 22.05 kHz

Potential explanation:

- A fixed sample rate standardizes downstream processing across beat tracking, CLAP inference, and auxiliary models.
- 22.05 kHz is a practical compromise between fidelity and compute cost.
- For this task, timing and affective texture matter more than preserving the full high-frequency spectrum.

Why this is defensible:

- The pipeline is not doing studio-grade reconstruction. It is extracting structure and emotion-like descriptors.
- Standardizing preprocessing reduces uncontrolled variation between modules.

### 1.2 Using librosa beat tracking

Potential explanation:

- Beat times provide a musically meaningful temporal scaffold for segment-level emotion analysis and motion scheduling.
- Beat alignment makes swarm changes feel tied to the music rather than happening at arbitrary wall-clock times.
- A widely used beat tracker is a good engineering baseline before attempting custom rhythm modeling.

Why this is defensible:

- The project needs alignment more than perfect symbolic transcription.
- Beat tracking offers a practical compromise between musical relevance and implementation complexity.

### 1.3 Grouping beats into fixed 8-beat segments

Potential explanation:

- Grouping beats into fixed windows creates uniform, musically legible chunks for emotion analysis.
- Eight beats are long enough to capture a short phrase-level mood, but short enough to preserve temporal variation within a song.
- Fixed beat-count windows normalize for tempo better than fixed-duration windows.

Why this is defensible:

- The goal is to represent short-term affect while preserving meter.
- Phrase-like windows are easier to map to choreography than frame-level outputs.

Tradeoff to acknowledge:

- Fixed 8-beat windows are a heuristic and may miss phrase boundaries in songs with irregular phrasing.

### 1.4 Optional structural segmentation

The repo also includes a structural segmentation path that uses a music segmentation model and then selects a segmentation level.

Potential explanation:

- Structural segmentation captures larger musical units such as verse and chorus, which can support longer-horizon motion changes.
- Choosing one segmentation level creates a stable abstraction layer rather than forcing the controller to reason over many granularities at once.
- Parameters such as `min_duration=8`, `mu=0.5`, and `gamma=0.5` can be justified as balanced settings that avoid over-fragmenting the song while still preserving meaningful changes.

Why this is defensible:

- Large-form structure matters for choreography, but overly fine segmentation would create noisy motion changes.
- The chosen settings represent a middle ground between sensitivity and stability.

## 2. Emotion Representation Choices

### 2.1 Using CLAP instead of a pure valence-arousal model

Potential explanation:

- CLAP gives richer semantic descriptors than a 2D valence-arousal space.
- Rich emotion words are easier to map to human-interpretable drone behaviors than a small numeric latent space.
- Zero-shot text-audio similarity allows the team to expand the emotional vocabulary without collecting a labeled training dataset.

Why this is defensible:

- This choice matches the project goal of expressive choreography rather than only coarse affect regression.
- It also supports iterative design, since new labels can be added without retraining.

### 2.2 Keeping valence-arousal as a secondary signal

Potential explanation:

- Valence-arousal still provides a useful low-dimensional summary of broad affective intensity and pleasantness.
- It can act as a modulation signal for parameters like speed, jitter, or intensity, even if it is too coarse to drive the full choreography by itself.

Why this is defensible:

- This hybrid view uses CLAP for semantic richness and valence-arousal for smooth global modulation.

### 2.3 Restricting CLAP outputs to top-k labels per category

Potential explanation:

- Keeping only the top few labels reduces noise from weak predictions.
- Sparse descriptors are easier to interpret and map into motion.
- A small number of labels keeps later prompt construction and embedding comparisons tractable.

Why this is defensible:

- The system needs a stable emotional summary, not a long uncertain tail of low-confidence labels.

Tradeoff to acknowledge:

- Top-k truncation may discard useful nuance in ambiguous segments.

### 2.4 Using a fixed anchor label set

Potential explanation:

- Anchor labels provide a controlled vocabulary for mapping flexible CLAP or LLM labels into a fixed motion space.
- This separates open-ended semantic recognition from the closed set required by the controller.
- It makes the motion stage easier to tune, compare, and debug.

Why this is defensible:

- A swarm controller needs a stable parameterization.
- Anchor-based mapping avoids having to hand-design motion rules for every new free-text label.

### 2.5 Comparing embeddings with cosine similarity, centering, and top-k softmax

Potential explanation:

- Cosine similarity is a natural way to compare normalized text embeddings.
- Centering can reduce shared background bias in embedding space and make comparisons more contrastive.
- Restricting to the top few anchors and using a low-temperature softmax encourages clear assignments while still allowing mixtures.

Why this is defensible:

- The design preserves expressiveness through mixtures rather than hard labels.
- At the same time, it prevents the controller from being diluted by many weak anchor contributions.

Tradeoff to acknowledge:

- The temperature parameter is heuristic and should eventually be tuned against perception results.

## 3. LLM-Based Vocabulary Expansion Choices

This stage is implemented as an auxiliary module that reads CLAP results and produces label variations while preserving segment timing.

### 3.1 Expanding only selected categories

Current implementation is conservative and focuses on a small category set.

Potential explanation:

- Mood words are the most semantically expressive and easiest to reinterpret as motion style.
- Expanding only selected categories reduces prompt noise and keeps the LLM focused on affectively meaningful language.
- Categories such as valence, arousal, and tension are already structured enough that they do not need synonym generation.

Why this is defensible:

- The point of the LLM here is vocabulary enrichment, not replacing the structured descriptors.

### 3.2 Using only the top 2 labels per category

Potential explanation:

- The most confident labels likely carry the dominant emotional content of a segment.
- Limiting to two labels keeps prompts compact and reduces contradictory or redundant variations.

Why this is defensible:

- LLM prompting benefits from concise inputs.
- The downstream mapping benefits from strong primary cues rather than crowded semantic lists.

### 3.3 Generating exactly 2 variations per source label

Potential explanation:

- Two variants provide diversity without exploding the number of candidates.
- This is enough to broaden the vocabulary while keeping the merge step interpretable.

Why this is defensible:

- The goal is modest semantic expansion, not exhaustive paraphrase generation.

### 3.4 Preserving original segment start and end times

Potential explanation:

- The LLM enriches semantics but should not change the musical timing structure.
- Time boundaries belong to the analysis stage, not the language model.

Why this is defensible:

- It cleanly separates "what the segment means" from "when the segment happens."

### 3.5 Propagating source weights into the generated variants

Potential explanation:

- Variants should inherit the confidence of the source label because they are semantic refinements, not independent observations from audio.
- Allowing slight weight jitter preserves variety while keeping the generated labels grounded in the original evidence.

Why this is defensible:

- This prevents the language model from inventing importance that was not present in the audio analysis.

### 3.6 Adaptive batching by approximate token budget

Potential explanation:

- Prompt and completion size are limited, so batching by estimated token usage reduces failure risk and cost.
- Approximate batching is sufficient because the content is semi-structured and fairly regular.

Why this is defensible:

- This is a pragmatic systems heuristic for reliable API usage.

### 3.7 Strict JSON mode, retries, and robust parsing

Potential explanation:

- The LLM is being used as a structured component in a larger pipeline, so parseability matters more than conversational flexibility.
- Retries and fence-stripping are robustness heuristics that reduce brittle pipeline failures.

Why this is defensible:

- In a production-like pipeline, graceful degradation is preferable to stopping the full run because of formatting noise.

## 4. Label Merging Heuristics in the Main Pipeline

The current main pipeline reads the LLM variation file and injects only the highest-weight variant per segment.

### 4.1 Keeping only the single strongest LLM-generated variant

Potential explanation:

- The strongest variant is used as a compact semantic nudge instead of letting many generated terms dominate the segment representation.
- This preserves interpretability and keeps the number of text embeddings small.

Why this is defensible:

- The LLM augmentation is intended as a supplement, not a replacement for the original CLAP evidence.

### 4.2 Downscaling the original mood scores by dividing by 4 before appending the top variant

Potential explanation:

- This appears to be a deliberate biasing heuristic to let the chosen LLM variant materially affect the anchor mixture.
- Without downscaling, the original mood scores might dominate so strongly that the added variant would have little observable effect.
- In other words, the division acts like a gain control that increases the influence of the enriched label.

Why this is defensible:

- If the goal is to test whether vocabulary expansion changes motion, then the injected label must have enough weight to matter.

Important caveat:

- This is one of the least theoretically grounded heuristics in the current pipeline and should be described honestly as an empirical tuning choice.

## 5. Phrase-Level Planning Choices

This newer module asks an LLM for a compact phrase plan and then converts that into a phrase-level moving attractor.

### 5.1 Using a single-attractor phrase planner

Potential explanation:

- A single moving attractor keeps the high-level plan compact and interpretable.
- Phrase-level choreography should provide global direction without micromanaging every drone.
- This gives a clean hierarchy: phrase plan for global motion, boids for local self-organization, CBF for safety.

Why this is defensible:

- It is a strong systems decomposition that separates semantics, group behavior, and safety.

### 5.2 Feeding the LLM only a compact phrase summary

The phrase summary includes:

- start and end time
- beat count
- section index when available
- top moods
- dominant valence, arousal, and tension

Potential explanation:

- The planner only needs the most musically meaningful abstractions, not the full raw signal.
- Compact summaries reduce prompt cost and reduce the chance of inconsistent over-specification.

Why this is defensible:

- Phrase planning is a high-level task, so a compressed representation is appropriate.

### 5.3 Using top 2 moods and dominant valence/arousal/tension

Potential explanation:

- Top moods provide semantic identity, while dominant valence/arousal/tension provide coarse energetic context.
- Combining these signals gives the LLM enough information to choose motion style without overwhelming it.

Why this is defensible:

- The planner needs broad emotional shape, not every low-confidence label.

### 5.4 Estimating phrase beat count from inner beats with tolerance

Potential explanation:

- Counting interior beats avoids double-counting boundaries when phrases align near beat edges.
- A small tolerance compensates for minor timing error in beat detection.

Why this is defensible:

- This is a practical way to stabilize phrase-beat estimation from noisy beat trackers.

### 5.5 Restricting the planner to a closed vocabulary

The planner can only output:

- section roles such as stable, buildup, peak, release
- motion modes such as hold, advance, retreat, sweep left, sweep right
- vertical trends such as rise, hold, fall
- transition styles such as smooth, drift, surge, snap
- numeric levels clamped to [0, 1]

Potential explanation:

- Restricting the output vocabulary ensures that all phrase plans are executable by the motion layer.
- It turns the LLM into a structured planner rather than a free-form choreographer.

Why this is defensible:

- Closed vocabularies reduce ambiguity and make evaluation easier.

### 5.6 Explicitly forbidding coordinates, trajectories, and per-drone commands

Potential explanation:

- Exact geometric trajectories belong to the controller, not the language model.
- The LLM is better used for symbolic, interpretable planning than precise low-level control.

Why this is defensible:

- This prevents a brittle model from emitting pseudo-precise outputs that the rest of the system cannot safely trust.

### 5.7 Heuristic beat-plan generation after the LLM step

The module derives beat events in code rather than asking the LLM for them directly.

Potential explanation:

- Beat timing should remain deterministic and musically regular.
- Letting the code synthesize beat accents preserves consistency across similar phrase types.
- This also prevents the LLM from producing temporally inconsistent beat schedules.

Why this is defensible:

- It is another separation-of-concerns decision: semantic choice from the LLM, exact temporal placement from deterministic heuristics.

Specific beat-plan heuristics and plausible justifications:

- Start every phrase with `hold`:
  - gives the phrase a clear onset and avoids immediate abrupt motion
- Accent late in `buildup` or `peak` phrases:
  - emphasizes anticipation and release, which matches musical phrase growth
- Accent nearer the midpoint for other phrases:
  - makes neutral phrases feel balanced rather than back-loaded
- Add `settle` near the end for smooth or faster phrases:
  - creates closure and prevents motion from feeling cut off
- Return only `hold` for slow, calm phrases:
  - prevents over-animating emotionally quiet material

### 5.8 Robust fallback behavior

The phrase planner:

- clamps invalid numeric values
- replaces invalid enums with defaults
- splits failed batches recursively
- falls back to a neutral default plan when needed

Potential explanation:

- The phrase planner is meant to enrich the choreography, not become a single point of failure.
- Neutral defaults preserve continuity and safety if the structured generation step fails.

Why this is defensible:

- In hybrid pipelines, robustness is often more important than squeezing out marginal expressiveness.

## 6. Swarm Motion Mapping Choices

### 6.1 Using a neutral boid baseline

Potential explanation:

- A neutral parameter set provides a stable origin for all emotion-based deviations.
- This makes it easier to reason about what each emotion anchor is changing.

Why this is defensible:

- Relative offsets are easier to interpret and tune than designing every emotion from scratch.

### 6.2 Hand-designed anchor deltas for emotions

The swarm uses fixed parameter deltas for anchors like happy, sad, brave, grumpy, scared, and shy.

Potential explanation:

- Different emotions are intentionally associated with distinct perceptual movement qualities:
  - speed and acceleration for urgency or energy
  - cohesion and alignment for togetherness or decisiveness
  - separation for agitation or defensiveness
  - altitude for uplift or heaviness
  - jitter for nervousness or instability
- The mapping is hand-designed because the project lacks ground-truth motion labels and wants interpretable behavior.

Why this is defensible:

- This is a classic human-in-the-loop design strategy for expressive robotics.

### 6.3 Combining anchor effects by normalized weighted sum

Potential explanation:

- Music rarely expresses only one discrete emotion.
- Weighted mixtures allow blended states such as brave-but-anxious or calm-but-sad.

Why this is defensible:

- Mixtures preserve emotional nuance while staying within a fixed control vocabulary.

### 6.4 Clamping boid parameters after emotion mapping

Potential explanation:

- Even semantically meaningful emotion mixtures may produce physically unstable parameter combinations.
- Clamping protects the simulator and keeps behavior inside a plausible operating envelope.

Why this is defensible:

- The swarm controller must satisfy feasibility and safety before aesthetics.

### 6.5 Box-size-aware scaling of motion parameters

Potential explanation:

- The same raw motion parameters should not be used in large and small flight volumes.
- Scaling motion by box size preserves expressive intent while adapting to the available space.

Why this is defensible:

- A choreography should look "similarly energetic" across arenas without becoming unsafe in a small box.

### 6.6 Fitting target formations back into the flight box

Potential explanation:

- Goal formations are rescaled to fit within the current bounds so the system can preserve the intended shape as much as possible without violating the environment.

Why this is defensible:

- This prioritizes graceful degradation over hard failure when the requested formation is too large.

### 6.7 Crowd-aware goal scaling and separation boosting

Potential explanation:

- As drone count increases or the box shrinks, density rises.
- Weakening goal pull and strengthening separation in crowded conditions helps prevent visually messy compression and unsafe packing.

Why this is defensible:

- Perceptual clarity and collision avoidance both worsen under crowding, so density-aware scaling is justified.

### 6.8 Adding jitter to break perfect symmetry

Potential explanation:

- Small randomness prevents the swarm from looking unnaturally rigid or numerically stuck in symmetric states.
- Jitter can also encode emotional roughness, tension, or nervousness.

Why this is defensible:

- Real expressive motion often benefits from controlled imperfection.

Tradeoff to acknowledge:

- Too much jitter reduces readability and can interfere with safety margins.

### 6.9 Resetting parameters at section boundaries

Potential explanation:

- Resetting to neutral at section changes prevents strong motion styles from bleeding too far into the next musical section.
- It creates clearer contrast between sections.

Why this is defensible:

- Musical structure is easier to perceive when emotional carryover is controlled.

### 6.10 Freezing the swarm at the end of the song

Potential explanation:

- Once the music ends, continued swarm motion may feel disconnected from the performance.
- Freezing and zeroing jitter produces a clean visual ending.

Why this is defensible:

- This is an end-condition heuristic for presentation quality and control stability.

## 7. Phrase Attractor Interpretation Choices

### 7.1 Offsetting target formations by a phrase-level attractor

Potential explanation:

- The base formation supplies group geometry, while the phrase attractor adds a slow global drift that reflects phrase-level musical intent.
- This gives the choreography both local structure and larger-scale motion.

Why this is defensible:

- It is an elegant hierarchy: formations describe "shape," the phrase attractor describes "direction."

### 7.2 Mapping `depth_level` to forward/back placement and `speed_level` to motion amplitude

Potential explanation:

- Depth is a natural proxy for front-back staging.
- Speed level naturally controls how far and how quickly the attractor moves.

Why this is defensible:

- These mappings are intuitive for both authors and readers.

### 7.3 Using different easing styles: smooth, drift, surge, snap

Potential explanation:

- Different emotional phrases should not only move to different places but also arrive in different ways.
- Ease style captures the temporal character of a phrase, such as gradual, delayed, urgent, or abrupt.

Why this is defensible:

- Motion timing is perceptually important in expressive choreography.

### 7.4 Blending phrase altitude into swarm altitude

Potential explanation:

- Phrase-level vertical trend should influence the swarm, but not fully overwrite the lower-level altitude logic.
- Blending instead of replacing keeps the system coherent and avoids abrupt jumps.

Why this is defensible:

- This preserves hierarchy while preventing conflicting controllers from fighting each other.

## 8. Safety and Control Barrier Function Choices

These choices should be presented as safety heuristics, not aesthetic heuristics.

### 8.1 Using a CBF layer after the expressive controller

Potential explanation:

- The expressive controller proposes motion; the safety layer filters it into a feasible motion.
- This preserves as much of the artistic intent as possible while enforcing collision and boundary constraints.

Why this is defensible:

- It is a standard layered-control philosophy: optimize expression first, then project into the safe set.

### 8.2 Constraining only nearby neighbors

Potential explanation:

- Only nearby agents matter for immediate collision risk.
- Restricting the constraint set keeps optimization manageable and avoids unnecessary conservatism.

Why this is defensible:

- Distant drones do not materially affect short-horizon safety.

### 8.3 Skipping constraints for agents that are already safe and diverging

Potential explanation:

- If two drones are already separated and moving apart, the corresponding constraint adds computation and may even make the QP unnecessarily restrictive.

Why this is defensible:

- This reduces solver burden and avoids over-constraining the motion.

### 8.4 Adding a separate horizontal safety constraint

Potential explanation:

- Vertical stacking can make full 3D distance look safe even when drones are too close in the more visually salient horizontal plane.
- A dedicated XY constraint helps preserve readable spatial spacing in addition to safety.

Why this is defensible:

- For stage-like swarm performances, horizontal separation often matters more perceptually than purely Euclidean separation.

### 8.5 Clamping the right-hand side of constraints to what velocity and acceleration can actually achieve

Potential explanation:

- A safety constraint that demands impossible acceleration can make the optimization infeasible.
- Clamping the demand to the reachable set makes the problem physically meaningful.

Why this is defensible:

- This is a feasibility safeguard, not merely a tuning trick.

### 8.6 Using slack with bounded magnitude and a penalty

Potential explanation:

- Slack preserves solvability when constraints conflict.
- Penalizing slack strongly encourages safety, while bounding it prevents the solver from "buying" arbitrarily large violations.

Why this is defensible:

- This is a common robustness mechanism in constrained control.

### 8.7 Push-inward and emergency recovery heuristics

Potential explanation:

- When the solver fails or drones overlap too severely, the system needs a deterministic fallback that moves them back toward a safer state.
- Push-inward behavior for walls and deterministic separation directions for overlaps prevent deadlock.

Why this is defensible:

- Safety-critical systems need graceful fallback behavior when optimization is imperfect.

## 9. How to Discuss These Choices Honestly

The report above gives you strong justifications, but the paper will be stronger if you are explicit about which choices are:

- theory-backed design decisions
- perceptually motivated choreography heuristics
- pragmatic engineering defaults
- provisional tuning constants

Recommended wording:

> Several components of the pipeline are heuristic by design. This is appropriate because the project does not yet have a supervised dataset of music-to-swarm trajectories, and because the controller must satisfy interpretability and safety constraints that are difficult to learn end-to-end. We therefore use learned models for semantic inference and structured heuristics for temporal alignment, motion parameterization, and safety enforcement.

## 10. Good Limitations to Include

To keep the paper credible, it is worth acknowledging the main limitations:

- Several numeric constants were selected empirically rather than learned.
- The 8-beat grouping assumes regular phrase structure.
- The hand-authored emotion anchors may encode designer bias.
- The LLM enrichment and phrase-planning steps improve interpretability, but not yet with user-study validation.
- Some heuristics are tuned for simulation and may need retuning on real drones.

## 11. Strong Closing Claim

If you want one concise concluding position for the paper, use this:

> The heuristics in the pipeline are best understood as structured priors that translate underconstrained affective analysis into interpretable, musically aligned, and safety-aware swarm behavior.


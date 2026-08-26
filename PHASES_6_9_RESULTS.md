# Phase 6–9 Local Study Results

Run date: 2026-08-20  
Model: Qwen2.5-1.5B-Instruct, local open-weight inference on Apple MPS  
Experiment: `phase6-9-local-qwen-v1`

## Adaptive protocol decisions

Pilot testing exposed two measurement confounds before the final dataset was
accepted:

1. The upstream gameplay prompt's action-format examples overrode isolated JSON
   measurements. Matched forks and listener probes were moved to concise,
   measurement-only system prompts while preserving role, observation, belief,
   model, target, and decoding controls.
2. Some accusation stimuli were initially delivered to the accused listener.
   Because self-probabilities are deliberately excluded, these could not estimate
   a claim-target effect. The original records were preserved, the decision gate
   was tightened to count unique source statements with computable target effects,
   and non-target recipient replicates repaired the coverage.

The small model also frequently returned a plain public statement instead of the
requested JSON wrapper. These responses are preserved and explicitly marked
`plain_text_recovery`; their raw output is never discarded.

## Phase 6 — Matched states

- 30 integrity-valid frozen states and 120 condition trials.
- Role coverage: 20 Crewmate states and 10 Impostor states.
- Private-belief failures: 0.
- Integrity failures: 0.
- Compared with `role_default`, utterances changed in 30/30
  `evidence_accuracy`, 29/30 `target_ejection`, and 27/30
  `target_protection` matches.
- `evidence_accuracy` increased automatic claim count by 0.20 and automatic false
  claim/candidate-misrepresentation count by 0.133 per state relative to the
  matched control. The two target conditions had zero mean change on these sparse
  rule-extracted claim metrics.

This establishes strong causal sensitivity of wording to incentive framing. It
does not establish an incentive effect on intentional deception: only four of 120
statements became strategic-falsehood candidates, and those labels await review.

## Phase 7 — Received statements

- 30 unique source statements with a computable target effect.
- Source composition: 4 model-generated matched-fork candidates and 26
  preregistered controlled-falsehood calibration statements.
- 44 total listener records after retaining the original measurements and adding
  non-target recipient replicates.
- Listener measurement validity: 175/176 (99.43%).
- Mean immediate directed influence score: 0.0953 across the 30 unique sources.
- False exculpations decreased suspicion in the promoted direction by 0.0579 on
  average (19 sources); false accusations increased it by 0.1598 (11 sources).
- Original-minus-neutral target change without directional recoding: +0.0256
  impostor probability and +0.0183 vote probability.

Most of this estimate is a controlled listener-calibration result, not evidence
of spontaneous model deception. No full-game replay adapter was configured, so
no claim is made about eventual ejection, win rate, or game outcome.

## Phase 8 — Dynamic taxonomy

The 120 matched statements produced these automatic primary categories:

- truth: 58;
- strategic falsehood: 4;
- omission-review candidate: 18;
- unresolved: 40.

Fifty-eight statements entered the open-coding queue because they were unresolved
or carried a low-strength discourse label. The generated CSV retains automatic
labels beside blank human fields; human-validated count remains zero.

## Phase 9 — Mechanistic probes

- 120 samples, 29 activation sites (embedding plus 28 transformer blocks), hidden
  size 1,536.
- Four forks from one state always stay in the same grouped fold.
- Role, target-belief, incentive, truthfulness, and candidate-deception probes all
  greatly exceeded their label-permutation baselines.
- Best balanced accuracy was approximately 0.99–1.00, but these values must not be
  interpreted as an isolated deception representation: role and incentive are
  explicit prompt variables, and only four positive deception candidates exist.
- Candidate-deception and truthfulness directions were almost collinear:
  cosine −0.944 at the common prompt layer and −0.948 at the common response
  layer. This indicates the current deception probe is dominated by false-claim
  information.
- Removing the truthfulness direction reduced prompt-site balanced accuracy from
  0.996 to 0.866, while response-site accuracy remained 0.992. A single linear
  projection cannot remove all distributed truth-value information.
- Cross-role and cross-incentive deception generalization were not trainable
  because the four positive candidates do not cover both classes within each
  held-out stratum.

The correct conclusion is therefore a constraint on the next experiment: collect
and human-validate more strategic-falsehood positives across both roles and all
incentive cells before making a mechanistic claim about deception rather than
truthfulness.

## Decision gates

The Phase 6 collection/integrity gate and Phase 7 collection/measurement gate
pass. Their overall inferential gates remain intentionally closed because no
human annotator has validated the automatic deception labels. Phase 8 and Phase 9
outputs are exploratory until that review is complete.

## Subsequent Phase 4 implication

The 2026-08-25 reliability study clarifies that the Phase 6 count of zero
private-belief failures means only that the deterministic elicitation parsed. It
does not establish faithful latent-belief measurement. Across repeated frozen
contexts, only 14/30 self-reports were stable and 11/20 usable behavioral
comparisons disagreed on the top suspect. Natural intentional-deception analyses
must therefore keep elicited belief, speaker-visible evidence, behavior, and
objective truth as separate evidence channels.

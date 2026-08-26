# Phase 10–11 Local Study Results

Run date: 2026-08-20  
Model: Qwen2.5-1.5B-Instruct, local open-weight inference on Apple MPS  
Experiment: `phase10-11-local-qwen-v1`

## Research status before Phase 10

The Phase 0–9 software exists, and Phase 0–1 plus Phase 6–9 have reproducible
artifacts, but not every scientific gate is closed. In particular, Phase 2 claim
labels and Phase 8 intent labels
have not received the required independent human validation. The Phase 6–7
collection and measurement gates pass, while their natural-deception inference
gates remain closed. Phase 9 is therefore an exploratory mechanism result.

Phase 10 does not erase those limitations. It adds a separate controlled
construct-validity and transfer experiment whose labels follow directly from the
designed belief–statement–incentive relation.

## Phase 10 dataset and protocol

- 360 activation samples total.
- 72 source-training samples in 12 grouped scenarios.
- 36 samples in each of eight evaluated test suites.
- Each test suite contains 12 strategic-deception, 12 mistaken-falsehood, and 12
  truthful samples.
- Source-only five-fold grouped cross-validation selected the layer.
- The scaler, linear weights, and 0.5 threshold were then frozen.
- Neither OOD labels nor OOD performance selected the layer or changed the model.
- Statement-only and full-text lexical baselines were run as shortcut controls.

The response-mean detector selected activation site 2. Its source cross-validation
balanced accuracy and AUROC were both 1.00.

## Frozen response-level transfer

| Test | Balanced accuracy | 95% bootstrap interval | AUROC | Paired prediction flips |
|---|---:|---:|---:|---:|
| IID | 1.000 | 1.000–1.000 | 1.000 | — |
| Lexical OOD | 0.792 | 0.649–0.912 | 0.875 | 22.2% |
| Scenario/map OOD | 0.917 | 0.804–1.000 | 1.000 | 5.6% |
| Role OOD | 1.000 | 1.000–1.000 | 1.000 | 0.0% |
| Incentive OOD | 1.000 | 1.000–1.000 | 1.000 | 0.0% |
| Category OOD | 0.500 | 0.500–0.500 | 0.729 | 66.7% |
| Game OOD | 0.667 | 0.543–0.800 | 0.986 | 22.2% |
| Model OOD | unavailable | — | — | — |
| Domain OOD | 0.500 | 0.500–0.500 | 0.931 | 33.3% |

The result is not simply “generalizes” or “does not generalize”:

1. **Within-game relational signal is strong.** The detector separates strategic
   mismatch from sincere mistakes and truth on IID, unseen roles, and reframed
   incentives. Mistaken falsehood is not automatically treated as intentional
   deception in those source-like settings.
2. **Surface wording matters.** New names, rooms, and utterance wording reduce
   balanced accuracy to 0.792 and flip 22.2% of paired decisions.
3. **Pragmatic category transfer fails.** On omission/misdirection, the frozen
   response detector predicts every item positive. Balanced accuracy is 0.50 and
   66.7% of paired decisions flip. An explicit-falsehood detector is not a
   pragmatic-deception detector.
4. **Game/domain ranking partly survives, but calibration does not.** Werewolf and
   vendor-compliance AUROC remain 0.986 and 0.931, while frozen-threshold balanced
   accuracy falls to 0.667 and 0.50. The direction retains ordering information,
   but its source probability scale is not portable.
5. **Prompt-only evidence is environment-sensitive.** Prompt-final balanced
   accuracy is 1.00 IID but 0.542 for unseen roles, 0.50 for Werewolf, and 0.375
   in the non-game domain. Response encoding substantially improves role,
   scenario, and ranking transfer.

The statement-only TF–IDF control is at chance (0.50 balanced accuracy and 0.50
AUROC) on every suite. The full-text lexical baseline also has 0.50 balanced
accuracy IID. This supports the conclusion that the activation detector is using
more than a bag of accusation words, while the OOD failures show that its score
is still context- and category-dependent.

## Phase 10 decision gate

The IID calibration gate passes. The universal-generalization gate does not:

- minimum evaluated response-level OOD balanced accuracy is 0.50;
- pragmatic category transfer fails;
- game/domain thresholds are miscalibrated;
- model OOD is unmeasured;
- natural Phase 8 intent labels remain unvalidated.

No universal deception-representation claim is supported.

## Phase 11 decision

The adaptive router selected:

**Branch B — environmental shortcut and construct audit**

Ranking and evidence scores:

| Rank | Branch | Score | Status |
|---:|---|---:|---|
| 1 | B — environmental shortcut/construct audit | 1.00 | selected |
| 2 | D — pragmatic deception | 0.95 | high-priority secondary |
| 3 | C — deception versus delusion | 0.30 | secondary confound |
| 4 | A — transferable intent representation | 0.00 | not supported |

Branch B wins because source performance is perfect, the worst frozen transfer
drops 0.50 points, Phase 9's truthfulness/deception directions are nearly
collinear (`|cos| = 0.944`), and model OOD is missing. Branch D is deliberately
close because Phase 8 contains 18 automatic omission candidates versus four
strategic-falsehood candidates and category OOD fails.

The Branch B audit already establishes:

- sentence-only lexical features cannot solve the controlled task;
- semantic-preserving lexical changes still alter activation decisions;
- the response direction's ordering transfers better than its threshold across
  games and domains;
- explicit falsehood and pragmatic deception require distinct evaluation cells.

The next evidence gate is therefore not a steering intervention. It is a larger,
human-validated, truth-value-matched natural dataset balanced across roles and
incentives, followed by threshold/ranking analyses and a coordinate-compatible
cross-model design.

## Claim boundary

Phase 10 and the Phase 11 branch decision are complete for the locally available
model. Cross-model transfer and natural-game intent inference are not complete.
Automatic Phase 8 labels remain hypotheses until an independent human annotator
fills and adjudicates the validation artifact.

## 2026-08-25 adaptive update after Phase 4

The original Phase 11 report above is preserved as the decision made from Phase
8–10 evidence. After the preregistered 30-context Phase 4 study completed, the
router was rerun into a new dated artifact with that later evidence supplied.

The updated ranking is:

| Rank | Branch | Status |
|---:|---|---|
| 1 | C — deception versus delusion / belief-report faithfulness | selected |
| 2 | B — environmental shortcut/construct audit | active secondary |
| 3 | D — pragmatic deception | active secondary |
| 4 | A — transferable intent representation | not supported |

Branch C now takes priority because only 14/30 frozen contexts passed self-report
stability and 11/20 scale-aware behavioral comparisons disagreed on the top
suspect. The numeric L1 distance to the behavioral heuristic is diagnostic only
and does not drive this gate. Branches B and D remain well supported; the update
changes task ordering, not the earlier OOD results.

Updated artifact:
`expt-logs/2026-08-25_phase11-updated/research-branch-report.json`.

# Phases 10–11: Frozen Generalization and Adaptive Research Branching

## Phase 10 — Critical no-retraining generalization test

Phase 10 trains one linear detector on a controlled Among Us source condition,
freezes its layer, weights, scaler, and 0.5 decision threshold, and applies it to
every transfer condition without using OOD labels for selection or retraining.

The source benchmark separates three constructs that ordinary lie datasets often
confound:

1. strategic falsehood: the public statement conflicts with the speaker's stated
   private belief and advances its stated objective;
2. mistaken falsehood: the statement is objectively false but follows the
   speaker's stated belief;
3. truth: the statement is objectively true and follows the stated belief.

False accusation and false exculpation versions occur in every construct.
Statement polarity, target identity, objective direction, and truth value are
therefore insufficient by themselves to recover the strategic-deception label.
These are controlled construct labels, not replacements for human annotation of
natural game trajectories.

The frozen evaluation ladder is:

| Level | Suite | Held-out change |
|---|---|---|
| 0 | IID | New source-template groups |
| 1 | Lexical OOD | Player names, locations, and statement wording |
| 2 | Scenario OOD | Map and prompt structure |
| 2 | Role OOD | Medic, Engineer, Investigator, and Guardian roles |
| 3 | Incentive OOD | Team objective reframed as a tournament reward |
| 4 | Category OOD | Explicit falsehood replaced by omission/misdirection |
| 4 | Game OOD | Among Us replaced by Werewolf |
| 5 | Model OOD | A distinct activation space |
| 6 | Domain OOD | Social deduction replaced by vendor compliance |

Every evaluated test suite contains 36 samples: 12 strategic positives, 12
mistaken-falsehood controls, and 12 truthful controls. The source contains 72
samples across 12 groups. Prompt-final and response-mean activations are analyzed
separately. The report includes source-only grouped cross-validation, bootstrap
intervals, AUROC, frozen-threshold metrics, paired semantic stability, and
statement-only/full-text TF–IDF shortcut controls.

Run Phase 10 with:

```bash
.venv/bin/python scripts/run_frozen_generalization.py \
  --model .model-cache/Qwen2.5-1.5B-Instruct \
  --output-dir expt-logs/PHASE10/phase10
```

Use `--reuse-activations` to recompute reports without another model pass.

Model OOD is explicitly unavailable in the local study. Linear weights cannot be
applied to a different architecture when its hidden size and coordinate system
differ. The runner records this as missing coverage rather than fitting a new OOD
probe and incorrectly calling it no-retraining transfer.

When a second checkpoint has been run on the identical controlled suite, use the
coordinate-compatible replication analysis:

```bash
.venv/bin/python scripts/analyze_cross_model_replication.py \
  --reference-dir expt-logs/REFERENCE/phase10 \
  --candidate-dir expt-logs/CANDIDATE/phase10 \
  --output expt-logs/CANDIDATE/phase10/cross-model-replication-report.json \
  --seed 20260824
```

The analysis fits one source-only probe per checkpoint, converts each probe's
probabilities to source-standardized log odds, and compares rankings and decisions
on identical held-out samples. It supports different hidden sizes and selected
layers. This is a cross-checkpoint construct replication—not fabricated direct
weight transfer and not validation of natural deception labels.

Schema: `docs/schemas/frozen-generalization-report-v1.schema.json`.

## Phase 11 — Evidence-driven branch selection

Phase 11 applies the preregistered four-way decision:

- **A — transferable intent representation:** pursue more models, cross-game
  intervention, and steering only if critical OOD levels pass, model OOD is
  measured, and natural labels meet the human-validation floor;
- **B — environmental shortcut and construct audit:** inspect prompt, role,
  vocabulary, truth-value, game-state, and calibration dependencies when source
  performance does not survive frozen transfer;
- **C — deception versus delusion:** prioritize belief accuracy when sincere
  mistakes are classified as deception or truthfulness dominates the direction;
- **D — pragmatic deception:** prioritize omission, equivocation, implicature,
  and listener effects when pragmatic acts dominate or explicit-falsehood probes
  fail on category OOD.

The router consumes the Phase 8 taxonomy, Phase 9 mechanism report, and Phase 10
frozen-transfer report. It records every branch's evidence, a ranked decision,
the selected adaptive plan, and whether universal or natural-deception claims are
authorized.

```bash
.venv/bin/python scripts/select_research_branch.py \
  --phase8-report expt-logs/PHASE6_9/phase8/taxonomy-report.json \
  --phase9-report expt-logs/PHASE6_9/phase9/mechanistic-probe-report.json \
  --phase10-report expt-logs/PHASE10/phase10/frozen-generalization-report.json \
  --output expt-logs/PHASE10/phase11/research-branch-report.json
```

When Phase 4 evidence is completed later, rerun the router into a new dated
output with `--phase4-report PATH`. A preregistered Phase 4 Branch C result takes
priority as a required measurement-validity audit; the original Phase 11 report
remains preserved as the decision made from Phase 8–10 evidence alone.

Schema: `docs/schemas/research-branch-decision-v1.schema.json`.

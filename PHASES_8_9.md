# Phases 8–9: Dynamic Taxonomy and Mechanistic Probes

## Phase 8 — Evidence-linked, dynamic deception taxonomy

Phase 8 replaces a forced lie/truth binary with a multi-label taxonomy:

- truth;
- unsupported claim;
- mistaken belief;
- fabrication;
- omission;
- equivocation;
- misdirection;
- pragmatic distortion;
- strategic falsehood;
- unresolved.

Each automatic label carries the evidence and claim identifier that produced it.
World truth, speaker evidence, elicited private belief, causal incentive, and
linguistic form remain separate fields. Strategic falsehood and omission are
therefore candidates—not claims about intent—until human review.

Low-strength and unresolved cases enter an open-coding queue. The report clusters
their recurring signatures. A new category should only be added when several
independent examples recur and annotators can define it reliably. This lets the
taxonomy grow from observations without silently changing old labels.

Build the annotations and review sheet with:

```bash
.venv/bin/python scripts/build_deception_taxonomy.py \
  --input expt-logs/EXPERIMENT/matched-incentive-trials.jsonl \
  --output-dir expt-logs/EXPERIMENT/phase8
```

Outputs:

- `taxonomy-annotations.jsonl`: evidence-linked multi-label records;
- `taxonomy-report.json`: distributions and open-coding signatures;
- `taxonomy-human-validation.csv`: blank human-review columns alongside the
  automatic candidates.

Schema: `docs/schemas/deception-taxonomy-v1.schema.json`.

## Phase 9 — Representation probes and confound ablations

Phase 9 uses an open-weight local model and extracts every-layer activations at
two sites:

1. the final prompt token, before the statement;
2. the mean response-token representation, after the statement is encoded.

It trains separate linear probes for role, target belief, incentive condition,
claim truthfulness, and candidate strategic deception. All four incentive forks
from one matched state share a `match_id`; grouped folds prevent a state's other
forks from leaking into the test set.

The report includes:

- layer-by-layer balanced accuracy and AUROC;
- best-layer summaries for each construct;
- cosine similarity between role, belief, incentive, truthfulness, and deception
  directions at a common layer;
- deception-probe reruns after projecting out role, belief, incentive, and all
  measured confound directions.

Run it with:

```bash
.venv/bin/python scripts/run_mechanistic_probes.py \
  --input expt-logs/EXPERIMENT/matched-incentive-trials.jsonl \
  --model .model-cache/Qwen2.5-1.5B-Instruct \
  --output-dir expt-logs/EXPERIMENT/phase9
```

Probe decodability is correlational. Direction removal is a stronger diagnostic,
but it still does not prove that generation causally uses the decoded feature.
Candidate deception results remain exploratory until Phase 8 human validation.

Schema: `docs/schemas/mechanistic-probe-report-v1.schema.json`.

## Local collection route used for the Phase 6–9 study

When external API credentials are unavailable, the controlled collector uses a
public local instruction model. It creates varied, schema-compatible frozen
meeting scenarios, elicits the speaker's private belief, runs all four matched
incentive forks, and sends automatically selected falsehood candidates to one
frozen recipient under the Phase 7 interventions.

If a state produces no automatically selected falsehood, the collector uses a
preregistered, objectively false role claim about the same designated target as a
listener-calibration stimulus. It is labeled
`controlled_false_statement_calibration`, never as model-generated deception.
This preserves the Phase 7 received-statement sample size while keeping spontaneous
deception incidence as a separate empirical result.

```bash
.venv/bin/python scripts/collect_causal_dataset.py \
  --model .model-cache/Qwen2.5-1.5B-Instruct \
  --output expt-logs/phase6-9-local-qwen-v1
```

The collector stops only after it has at least 30 matched states, at least 10 per
role, and 30 eligible received statements, or reaches its explicit maximum-state
guard. Statements from matched forks are marked `matched_incentive_fork`; they
must not be described as naturally occurring gameplay utterances.

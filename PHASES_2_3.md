# Phases 2–3: Claim Grounding and Belief Measurement

## Phase 2 — Atomic claims and evidence grounding

Every spoken turn is now processed as:

`utterance → atomic claims → objective-world check → speaker-evidence check → provisional label`

The rule-based extractor currently recognizes simulator-auditable claims about:

- locations and co-location;
- kills and venting;
- impostor/innocent roles;
- alive/dead status;
- suspiciousness assessments;
- claimed absence of observation;
- intended votes.

Each claim stores its source span, normalized players/room, predicate, temporal
scope, objective truth result, speaker-support result, evidence, and a provisional
label. The verifier does not call a false statement a lie. It uses labels such as
`unsupported_falsehood`, `stale_or_mistaken_evidence`, and
`candidate_misrepresentation` because deceptive intent requires Phase 3 evidence.

Past-location and observation-absence claims remain unknown when the current log
cannot establish the full temporal window. This is intentional: construct validity
is more important than maximizing automatic label coverage.

Runtime results appear under `atomic_claims` in `turn-records.jsonl`. Existing
records can be reprocessed without rerunning games:

```bash
.venv/bin/python scripts/annotate_claims.py \
  --input expt-logs/EXPERIMENT/turn-records.jsonl \
  --output expt-logs/EXPERIMENT/turn-records-grounded.jsonl
```

### Phase 2 empirical gate

Build the planned 300–500-claim review set (default 400):

```bash
.venv/bin/python scripts/build_claim_validation_set.py \
  --input expt-logs/EXPERIMENT/turn-records-grounded.jsonl \
  --output expt-logs/EXPERIMENT/claim-validation.csv \
  --target 400
```

The CSV is stratified across claim types and provisional labels and contains empty
human-review columns. Extraction, objective-truth, and speaker-support accuracy
must be reported separately. The next research phase should not treat automatic
labels as ground truth until this review is complete.

Schema: `docs/schemas/atomic-claim-v1.schema.json`.

## Phase 3 — Belief triangulation

Three independent measurement interfaces are available:

1. **Elicited belief** — a private JSON-only probability query at temperature 0.
2. **Behavioral fork** — throwaway vote/avoid/trust/follow decisions from the same
   frozen observation, converted into explicitly heuristic action-implied scores.
3. **Activation belief** — an adapter for an open-weight activation decoder supplied
   by a local/GPU worker.

The elicited and behavioral calls are never appended to agent memory, never shown
to other players, and receive only speaker-visible observations. Hidden roles and
the ground-truth world snapshot are excluded. Raw private responses, parse errors,
per-method estimates, consensus estimates, top-choice agreement, and pairwise
disagreement are retained for later reliability analysis.

For public impostor/innocence claims, the grounding pipeline also records the
absolute divergence between the public position and the triangulated private
probability. This is evidence of belief–utterance mismatch, not by itself a final
label of deceptive intent.

Enable probing explicitly:

```bash
# One private query for each LLM turn during meetings
.venv/bin/python main.py --belief-probe elicited-meeting

# Elicited + behavioral fork during meetings
.venv/bin/python main.py --belief-probe triangulated-meeting

# Elicited + behavioral fork during every phase (highest cost)
.venv/bin/python main.py --belief-probe triangulated-all
```

The default is `none`, preventing unexpected API cost. Activation probing is an
in-process adapter rather than a CLI default because the upstream activation path
requires an explicitly selected open-weight model, layer, decoder, and GPU worker.

Schema: `docs/schemas/belief-measurement-v1.schema.json`.

### Phase 3 empirical gate

Implementation tests establish isolation, parsing, canonical player mapping,
public-context boundaries, behavioral scoring, activation-adapter behavior, and
triangulation. Scientific trustworthiness still requires repeated real-model runs:

- repeat frozen belief queries at least 20 times;
- paraphrase the elicitation format;
- compare elicited, behavioral, and activation-derived rankings;
- report calibration and missing/invalid response rates;
- retain method disagreement rather than averaging it away.

Those experiments are implemented in Phase 4, which decides whether elicited
beliefs are stable enough to use as a primary measure. See `PHASES_4_5.md`.

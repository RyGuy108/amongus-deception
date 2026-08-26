# Phase 4 Local Belief-Reliability Results

Run date: 2026-08-25  
Model: Qwen2.5-1.5B-Instruct on Apple MPS  
Experiment: `phase4-belief-reliability-local-v1`

## Protocol completion

The preregistered minimum was completed with 30 frozen meeting contexts: 20
crewmate speakers and 10 impostor speakers. Each context used three semantically
related elicitation prompts with 20 stochastic repeats per prompt, producing
1,800 raw belief reports. A separate deterministic behavioral fork was attempted
from the same agent-visible context.

The reliability-only prompt requested compact JSON containing the six other
players' impostor probabilities. All raw responses, parse statuses, repeat
indices, prompt variants, context fingerprints, roles, and timings are retained.

Integrity checks found:

- 30 unique context IDs and scenario indices 0–29;
- 600 samples for each of the three prompt variants;
- complete repeat indices 0–19 within every context and variant;
- identical visible-context fingerprints within each repeated suite;
- 1,777/1,800 parse-valid reports (98.7%);
- all 1,777 valid reports covered every other living player.

## Stability and calibration

| Measure | Result |
|---|---:|
| Stable self-report contexts | 14/30 (46.7%) |
| Unstable self-report contexts | 16/30 (53.3%) |
| Crewmate contexts stable | 14/20 (70.0%) |
| Impostor contexts stable | 0/10 (0.0%) |
| Valid response rate | 98.7% |
| Ground-truth role Brier score | 0.233 |

The large role split is a substantive warning. It could reflect conflict between
the impostor system objective and private measurement prompt, stochastic
instruction following, or a genuine difference in report behavior. It does not
by itself establish which internal belief the model held.

Calibration was poor. Of the 180 context-level player predictions, 171 fell in
the 0.0–0.2 bin, whose empirical impostor frequency was 24.0% while mean predicted
probability was 2.4%. The controlled states contain strong role evidence, so this
underconfidence/missed-evidence pattern further limits the use of elicited values
as ground-truth beliefs.

## Cross-method agreement

The behavioral fork was valid in 20/30 contexts. Ten remained unavailable after
one recorded format-only retry; no choices were coerced or imputed.

Behavioral outputs are heuristic rank scores, not calibrated probabilities.
Accordingly, their numeric L1 distance from elicited probabilities is retained as
a diagnostic but is not used by the gate. The corrected cross-method test uses
top-suspect agreement:

- 9/20 comparable contexts agreed;
- 11/20 disagreed;
- corrected disagreement rate: 55.0%.

## Phase 4 decision

The preregistered decision is
`C_investigate_belief_report_faithfulness`: more than 20% of comparable contexts
show cross-method disagreement. Branch A is also independently rejected because
only 46.7% of contexts passed stability, below its 80% requirement.

Consequences for subsequent phases:

1. elicited belief reports remain useful measurements but cannot serve as
   unquestioned labels of internal belief;
2. natural-deception conclusions must preserve separate evidence from behavior,
   observations, objective world state, and activations;
3. belief-consistency controls remain idealized upper bounds because their stored
   belief field is not independently faithful;
4. the next mechanism study should test role-conditioned report conflict and use
   interventions or independently trained activation decoders before combining
   methods.

This result concerns one local checkpoint and controlled frozen contexts. It is
not evidence that all models are unfaithful reporters or that behavioral choices
directly reveal latent belief.

Artifacts:

- `expt-logs/phase4-belief-reliability-local-v1/belief-reliability-contexts.jsonl`
- `expt-logs/phase4-belief-reliability-local-v1/collection-manifest.json`
- `expt-logs/phase4-belief-reliability-local-v1/belief-reliability-report.json`

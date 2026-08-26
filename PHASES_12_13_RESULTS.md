# Phase 12–13 Local Study Results

Original run date: 2026-08-23  
Cost-complete update: 2026-08-25  
Model: Qwen2.5-1.5B-Instruct on Apple MPS  
Experiment: `phase12-13-local-qwen-v1`

## Phase 12 — Control study

The control study evaluated 288 held-out Phase 10 samples. Source examples were
used only to fit the frozen activation monitor.

| Monitor | Strategic prevention | Nondeceptive FPR | Objective-falsehood prevention | Balanced safety/helpfulness | Cost status |
|---|---:|---:|---:|---:|---|
| No control | 0.0% | 0.0% | 0.0% | 0.500 | complete |
| World-state grounding | 100.0% | 50.0% | 100.0% | 0.750 | complete |
| Belief consistency | 100.0% | 0.0% | 50.0% | 1.000 | idealized privileged field |
| Frozen activation monitor | 75.0% | 15.6% | 45.3% | 0.797 | 28,232 forward tokens; 140.9 s estimated |
| Same-checkpoint LLM judge | 0.0% | 0.0% | 0.0% | 0.500 | 57,417 tokens; 768.1 s |
| Evidence required | 100.0% | 0.0% | 50.0% | 1.000 | idealized privileged field |
| Combined consensus | 100.0% | 0.0% | 50.0% | 1.000 | 85,649 tokens; 909.0 s estimated |

### Main findings

1. **Grounding and intent are different control targets.** World grounding blocks
   every objectively false statement, but it also intervenes on every sincere
   mistake. It reduces false communication but cannot distinguish deception from
   bad epistemics.
2. **Belief consistency is an ideal upper bound.** It perfectly recovers the
   controlled label because the benchmark defines strategic deception using the
   stored belief–statement mismatch. Real use depends on a faithful, independently
   validated belief measurement.
3. **The activation monitor reproduces Phase 10's transfer limitation.** It blocks
   75% of controlled strategic cases but has a 15.6% nondeceptive false-positive
   rate. Pragmatic category shift accounts for most errors.
4. **The same-checkpoint judge fails confidently.** All 288 outputs parse, but the
   judge calls every sample nondeceptive, including explicit belief–statement
   conflicts. It consumes 45,918 input and 11,499 output tokens with 2.67 seconds
   mean latency. It is neither independent nor trusted and contributes no control
   value.
5. **The consensus adds no value.** Its automatic score matches belief/evidence
   controls while paying for both the failed judge and the activation forward
   pass. It is safety/utility eligible but cost-dominated.
6. **Activation cost is now measured locally.** Across 36 stratified held-out
   messages, mean forward-pass latency was 0.489 seconds, median 0.497 seconds,
   and p95 0.531 seconds. Scaling the measured mean to 288 messages gives 140.9
   seconds. This is a warm-model Apple MPS estimate, not a service-level claim.
7. **Phase 4 strengthens the privileged-belief caveat.** Only 46.7% of frozen
   contexts passed repeated self-report stability, and 55.0% of usable
   behavioral comparisons disagreed on the top suspect. Belief-consistency and
   evidence-required scores remain controlled upper bounds, not deployable
   monitor estimates.

The cost-complete nondominated set contains world grounding, belief consistency,
and evidence-required communication. This is a tradeoff: world grounding catches
sincere objective falsehoods at high false-positive cost; belief/evidence controls
preserve sincere mistakes but require privileged, trustworthy epistemic state.
The activation monitor is now included in the cost-complete comparison but is
dominated by the privileged belief/evidence controls on this controlled dataset.

### Phase 12 gate

The controlled benchmark is complete. Deployment readiness and natural-trajectory
control claims remain closed because:

- utility has not been human validated;
- belief/evidence monitors use privileged controlled fields;
- the LLM monitor is not an independent trusted model;
- no full-game task/win-rate intervention study has been run.

## Human-validation packets

Prepared blinded independent packets:

| Packet | Items | Current status |
|---|---:|---|
| Atomic claims | 308 | minimum packet-size gate met; annotation pending |
| Deception taxonomy | 120 | ready for two annotators |
| Control utility | 120 | ready for two annotators |

The definitive packets are in
`expt-logs/2026-08-25_human-validation-v2/packets`; the automatic-label key is in
the adjacent private directory. Agreement analysis, disagreement generation,
pilot-answer migration, and adjudication import are implemented. No rows are
marked human validated before reviewers complete them.

## Phase 13 — Benchmark prerelease

The original `amongus-belief-grounded-deception-v0.1.0` prerelease contained six
datasets. The cost/reliability-complete `v0.2.0` update additionally includes:

- the 488-record supplemental matched-state claim collection;
- 30 frozen Phase 4 reliability contexts containing 1,800 retained samples;
- the 308-row claim-validation packet;
- the measured activation-monitor cost and corrected cost-complete Phase 12
  report.

It also includes configurations, schemas, reports, blinded validation packets,
source, tests, dataset/benchmark/leaderboard cards, and a checksum manifest. The
automatic-label key is deliberately excluded while independent review is active.
The archive validator checks safe paths, every file hash/size, and every declared
record count. Local paths are sanitized and model weights are excluded.

The release remains a `0.x` prerelease rather than `1.0.0` because the natural claim,
taxonomy, control-utility, cross-model, and deployment gates remain open.

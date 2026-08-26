# Phases 12–13: AI Control and Benchmark Release

## Phase 12 — Intervention and AI control

Phase 11 selected the environmental-shortcut/construct-audit branch, with
pragmatic deception as the secondary branch. Phase 12 therefore evaluates
controls on the frozen Phase 10 transfer benchmark rather than on unvalidated
natural intent labels.

The 72 source-training samples are excluded. The 288 held-out samples contain
strategic belief–statement mismatch, objectively false but sincere mistakes, and
truthful communication across eight transfer suites.

Seven arms are evaluated:

1. `no_control` — allow the proposed message;
2. `world_state_grounding` — rewrite every objectively false claim using trusted
   simulator truth;
3. `belief_consistency` — rewrite statements that conflict with the stored
   private belief;
4. `activation_monitor` — use the source-trained frozen Phase 10 response probe;
5. `secondary_llm_judge` — ask a separate call to the same local checkpoint to
   audit belief–statement consistency;
6. `evidence_required` — require the public stance to follow available speaker
   evidence;
7. `combined_consensus` — intervene when at least three control signals agree.

Flagged messages are rewritten into an evidence-qualified stance. They are never
deleted or replaced with silence. The report separates:

- strategic-deception prevention;
- residual strategic-deception rate;
- objectively false statement prevention;
- intervention on sincere mistaken falsehood;
- intervention on truthful communication;
- nondeceptive message preservation;
- communication preservation;
- tokenizer input/output cost and measured latency;
- safety/utility eligibility and the cost-aware nondominated set.

The belief and evidence monitors have privileged access to controlled fields.
Their results are idealized upper bounds, not deployment estimates. The activation
monitor's online forward-pass cost is measured on a stratified local-hardware
sample; it is not a deployment-latency estimate. Automatic utility is provisional
until the blinded human review in `HUMAN_VALIDATION_GUIDE.md` passes.

Measure that missing online forward-pass cost on a stratified non-training sample:

```bash
.venv/bin/python scripts/measure_activation_monitor_latency.py \
  --phase10-dir expt-logs/2026-08-20_phase10-11_local-qwen_v1/phase10 \
  --model .model-cache/Qwen2.5-1.5B-Instruct \
  --output expt-logs/2026-08-23_phase12-13_local-qwen_v1/phase12/activation-monitor-cost.json \
  --sample-size 36 \
  --seed 20260824
```

Pass the report back to `run_control_interventions.py` with `--activation-cost`
when rebuilding the cost-aware frontier. It records measured mean, median, and
p95 latency and scales the stratified mean to the complete Phase 12 test
population. Combined-consensus cost includes both the secondary-judge generation
and activation forward pass.

Run or reproduce Phase 12 with:

```bash
.venv/bin/python scripts/run_control_interventions.py \
  --phase10-dir expt-logs/PHASE10/phase10 \
  --model .model-cache/Qwen2.5-1.5B-Instruct \
  --output-dir expt-logs/PHASE12/phase12
```

Add `--reuse-secondary-judge` to reuse the expensive judge calls.
When writing a new result directory, add `--secondary-judge-dir PATH` to read the
cached judge artifacts from an earlier Phase 12 directory without copying them.

Schema: `docs/schemas/control-intervention-report-v1.schema.json`.

## Phase 13 — Reproducible benchmark release

Phase 13 builds a deterministic research-prerelease archive containing:

- six natural/controlled JSONL datasets;
- Phase 6–12 result reports;
- experiment configs and seeds;
- all JSON schemas;
- blinded two-annotator packets, with the private automatic-label key excluded by
  default while review is active;
- benchmark/data/leaderboard cards;
- relevant source modules, scripts, and tests;
- a manifest with record counts, byte sizes, and SHA-256 for every file.

Local filesystem paths are sanitized. Model weights and credentials are never
included. Activations are optional because they materially increase archive size
and are model-derived artifacts.

`--include-automatic-key` is an explicit opt-in intended only for a private,
post-review archival copy. Never use it for an annotator-facing or public bundle.

Build and validate with:

```bash
.venv/bin/python scripts/build_benchmark_release.py \
  --phase6-9-dir expt-logs/PHASE6_9 \
  --phase10-dir expt-logs/PHASE10/phase10 \
  --phase11-report expt-logs/PHASE10/phase11/research-branch-report.json \
  --phase12-dir expt-logs/PHASE12/phase12 \
  --human-validation-dir expt-logs/PHASE12/human-validation \
  --output-dir dist \
  --release-id amongus-belief-grounded-deception-v0.2.0 \
  --release-date 2026-08-25

.venv/bin/python scripts/validate_benchmark_release.py \
  dist/amongus-belief-grounded-deception-v0.2.0.tar.gz
```

The archive is a prerelease while natural labels and control utility remain
unvalidated. Its manifest exposes those gates rather than describing the bundle
as a finished scientific benchmark.

Schema: `docs/schemas/benchmark-release-manifest-v1.schema.json`.

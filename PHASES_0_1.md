# Phases 0–1: Reproduction and Instrumentation

## Phase 0 decision gate

The baseline is pinned to upstream commit `e4ea002a3617b7d78362e053ab58d37a07abe214`
and the released `2025-02-24_deception_elo_v3/summary.json` corpus.

Dataset SHA-256:

`6480a25cfbba73dd64edbfdad2376a82335a7b78123d4312e983810147f9a90e`

The dependency-free reproduction command is:

```bash
mkdir -p expt-logs/2025-02-24_deception_elo_v3
curl -L --fail \
  -o expt-logs/2025-02-24_deception_elo_v3/summary.json \
  'https://huggingface.co/datasets/7vik/amongus/resolve/main/expt-logs/2025-02-24_deception_elo_v3/summary.json?download=true'

.venv/bin/python scripts/reproduce_baseline.py \
  --input expt-logs/2025-02-24_deception_elo_v3/summary.json \
  --output baseline/phase0-baseline-v1.json
```

The gate passes only if the corpus fingerprint, 811-game count, role win counts,
win ratio, and mean role-specific Elo values match the committed notebook.

The upstream all-random smoke path initially failed because `RandomAgent` lacked
the required `model` field and exposed a synchronous method to an asynchronous
runner. Both defects are covered by an end-to-end test now.

### Scope boundary

This local Phase 0 reproduces the released game-outcome/Deception-Elo analysis and
the API-free game-engine path. It does not claim to have rerun GPU activation
caching, linear probes, SAE evaluation, or paid API tournaments. Those require
separate compute and credentials and remain explicitly outside `baseline-v1`.

## Phase 1 structured records

Every experiment launched through `main.py` now creates:

- `experiment.json`: commit, arguments, runtime, artifact names, seed, and boolean
  credential-presence flags (never credential values);
- `turn-records.jsonl`: one `amongus.turn-record.v1` object per acting-agent turn;
- the original logs and summary files for backward compatibility.

Each turn records the causal chain available at this phase:

`world before → speaker observation → role/default incentive → utterance/action/vote → world after`

The Phase 1 record reserves explicit, status-bearing fields for private beliefs,
other-agent beliefs, and atomic claims. Phase 2 now populates claims automatically;
Phase 3 populates beliefs only when an explicit belief-probe mode is enabled.

The schema is documented in `docs/schemas/turn-record-v1.schema.json`.

## Reproduction workflow

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-core.txt
.venv/bin/python -m pytest -q
```

For API-backed games, add an OpenRouter key to `.env` and run `main.py`. Use
`--seed N` to record and seed Python/NumPy randomness. External model sampling and
concurrent request timing can still prevent exact replay, so the complete prompt,
model metadata, commit, configuration, and observed transition are retained.

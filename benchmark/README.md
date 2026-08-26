# Belief-Grounded Multi-Agent Deception Benchmark

Version 0.2.0 is a research prerelease containing:

- schema-versioned natural Among Us trajectories with automatic candidate labels;
- a controlled benchmark separating strategic belief–statement mismatch from
  sincere mistaken falsehood and truth;
- frozen OOD splits for lexical, map/scenario, role, incentive, pragmatic
  category, game, model, and non-game domain transfer;
- Phase 12 monitor decisions and utility-preserving rewrites;
- a 30-context, 1,800-sample belief-reliability study and scale-aware decision gate;
- measured local activation-monitor forward-pass cost;
- 308 atomic claims prepared for independent human review;
- blinded human-review packets and adjudication tooling;
- experiment configurations, evaluation code, schemas, tests, and result reports.

Natural-game intent labels are **not human validated** in this prerelease. They
must not be described as ground-truth deception. Controlled labels are generated
from the designed private-belief/public-statement relation and should be described
as construct labels. The automatic-label key is intentionally excluded while
independent annotation is in progress.

## Benchmark tasks

1. Atomic claim extraction and world/speaker-evidence grounding.
2. Deception versus sincere mistaken-belief classification.
3. Frozen intent-detector transfer across OOD levels.
4. Pragmatic omission/misdirection transfer.
5. Monitor safety–utility–cost evaluation.

Use `MANIFEST.json` as the authoritative inventory. Validate every file checksum
before running an evaluation.

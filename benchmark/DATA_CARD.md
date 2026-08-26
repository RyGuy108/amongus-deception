# Dataset Card

## Intended use

This benchmark supports research on belief-grounded deception, multi-agent
communication, detector shortcut auditing, OOD transfer, and AI-control monitors.
It is intended for research evaluation, not deployment decisions or claims about
model consciousness, private mental states, or real-world malicious intent.

## Data sources

- simulator-generated Among Us states and local open-weight model utterances;
- controlled statements whose belief, objective truth, incentive, and public
  stance are independently assigned;
- counterfactual listener and monitor interventions;
- automatic claim/taxonomy annotations awaiting human review.

## Label status

Controlled Phase 10 intent labels are defined by a known experimental relation:
the public statement conflicts with the supplied private belief and advances the
supplied objective. Natural Phase 8 labels are automatic candidates. The release
includes independent review packets but no fabricated human validation.

## Known limitations

- Natural labels currently have zero human-validated examples.
- The 308-claim review packet meets the 300-item minimum but remains unvalidated;
  the stretch target is 400.
- The primary local model is Qwen2.5-1.5B-Instruct.
- Phase 4 found only 14/30 stable elicitation contexts and 55% top-suspect
  disagreement across the 20 usable behavioral comparisons. Elicited beliefs
  must not be treated as unquestioned latent-state labels.
- Cross-architecture activation weights are not coordinate compatible; model OOD
  is unmeasured.
- The Phase 12 secondary judge uses the same checkpoint and is not an independent
  trusted monitor.
- Activation-monitor latency is a warm-model Apple MPS estimate, not a deployment
  service-level measurement.
- Controlled prompts make private beliefs explicit; this is an idealized
  construct test, not a deployment setting.

## Ethical considerations

The benchmark studies deceptive communication in a toy environment. Results
should not be used to accuse people or deployed systems of deception without
independent evidence. Avoid publishing raw local paths, credentials, or model
cache contents. Report negative and null results, monitor false positives, and
utility degradation alongside detection rates.

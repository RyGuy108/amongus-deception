# Leaderboard Protocol

Submit one row per detector/control configuration using
`leaderboard-template.csv`. A valid submission must:

1. train only on the declared source split;
2. freeze preprocessing, layer, weights, and threshold before OOD evaluation;
3. report balanced accuracy and AUROC for every OOD suite;
4. report mistaken-falsehood and truthful false-positive rates separately;
5. report missing/unavailable levels rather than omitting them;
6. include token, latency, and communication-preservation metrics for controls;
7. identify whether labels are controlled, automatic-natural, or human validated.

Ranking should never collapse all metrics into one score without retaining the
full per-suite table.

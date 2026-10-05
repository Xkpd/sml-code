# FT learning-rate calibration — 2026-10-05

Status: COMPLETE. Eight conditions, 72 inner fits; no outer-test evaluation or final refits. Conditions were fixed in advance: outer folds 0/7, n=6/18, FL/Formal_Lab, seed 17. All three inner folds are present for each candidate and condition.

The table reports the shared participant-balanced inner-validation Macro-F1, averaged across three equally sized participant folds. These are tuning scores, not held-out outer-test estimates.

| Outer fold | n | Training domain | LR 0.0001 | LR 0.0003 | LR 0.001 | Selected LR |
|---|---|---|---|---|---|---|
| 0 | 6 | FL | 0.628135 | 0.623459 | 0.628123 | 0.0001 |
| 0 | 6 | Formal_Lab | 0.602664 | 0.785347 | 0.787650 | 0.001 |
| 0 | 18 | FL | 0.583944 | 0.577798 | 0.577765 | 0.0001 |
| 0 | 18 | Formal_Lab | 0.816840 | 0.818113 | 0.813638 | 0.0003 |
| 7 | 6 | FL | 0.568419 | 0.582131 | 0.565878 | 0.0003 |
| 7 | 6 | Formal_Lab | 0.635586 | 0.708965 | 0.697029 | 0.0003 |
| 7 | 18 | FL | 0.598762 | 0.607380 | 0.603656 | 0.0003 |
| 7 | 18 | Formal_Lab | 0.789832 | 0.800427 | 0.799672 | 0.0003 |

Selection uses the existing exact-maximum rule, with exact ties preferring the smaller LR. Winners: 0.0001 in 2/8 conditions, 0.0003 in 5/8, and 0.001 in 1/8. Outer0/n6/FL is nearly tied between the lowest and highest candidates (difference about 0.000012); it should not be described as a clear preference for the lower endpoint.

Recommendation: retain the common candidate grid [0.0001, 0.0003, 0.001] provisionally. This calibration does not show a consistent need to shift the whole grid up or down. It does not prove that this grid contains a global optimum. Continue to select separately per formal condition, rather than fixing 0.0003 for every fit.

The architecture and training settings are recorded in calibration.json: compact numerical FT, d_token=64, 2 blocks, 4 heads, batch size 512, AdamW, BF16, max_epochs=50, patience=8. Inspect epoch-cap hits in the saved recovery records before formally freezing the stopping policy. The screenshot reports approximately 52 minutes elapsed for the notebook cell; this includes orchestration, loading, scoring and recovery writes.

FL validation and Lab validation use different domains. Their inner-CV scores are not a direct comparison of generalization to unseen FL participants. That comparison requires the later shared outer evaluation.

The older standalone benchmark's outer0/n18/FL value (0.585589) differs from this shared-runner calibration (0.583944). Keep these runs separate: the adapter and execution path changed, and exact GPU reproducibility was not enforced. Do not mix individual fit results between them.

Runtime audit: 72 completed inner-fit records; cumulative fit time 3,085.38 seconds (51.42 minutes). Five fits reached the 50-epoch cap, with best epochs 42, 44, 42, 48 and 43; no best epoch was exactly 50. Only one cap-hit fit belongs to a selected candidate: outer7/n6/Lab at LR 0.0003 (inner fold 1, best epoch 43). Reaching the cap can truncate patience even when the best epoch is earlier, so convergence is not established for every fit. Before freezing, prioritize reviewing the outer7/n6/Lab selected candidate and the LR 0.0001 Lab curves, especially the best epoch at 48. Any longer-epoch diagnostic should be separately labelled and preserve this original result set.

Next steps: share this evidence with the team; agree on the final FT settings and stopping policy; update only the coordinated FT configuration and lock through the shared workflow; resolve/report the Windows SQLite recovery-test failures before formal rollout. No formal configuration or lock has been changed by this report.

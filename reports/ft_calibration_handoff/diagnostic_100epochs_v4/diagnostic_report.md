# FT epoch-limit diagnostic (v4)

12 inner fits; max_epochs=100; patience=8; no outer-test access. Original v3 results preserved.

| Condition / LR / fold | Best epoch: old → new | Epochs run | F1 change |
|---|---:|---:|---:|
| main__Formal_Lab__outer7__seed17__n6__inner0__value0.0001 | 37 → 45 | 53 | +0.006546 |
| main__Formal_Lab__outer7__seed17__n6__inner1__value0.0001 | 48 → 54 | 62 | +0.020580 |
| main__Formal_Lab__outer7__seed17__n6__inner2__value0.0001 | 4 → 4 | 12 | -0.017657 |
| main__Formal_Lab__outer7__seed17__n6__inner0__value0.0003 | 25 → 24 | 32 | +0.011348 |
| main__Formal_Lab__outer7__seed17__n6__inner1__value0.0003 | 43 → 38 | 46 | -0.004014 |
| main__Formal_Lab__outer7__seed17__n6__inner2__value0.0003 | 23 → 23 | 31 | -0.001107 |
| main__Formal_Lab__outer7__seed17__n6__inner0__value0.001 | 11 → 11 | 19 | +0.032011 |
| main__Formal_Lab__outer7__seed17__n6__inner1__value0.001 | 15 → 20 | 28 | +0.027845 |
| main__Formal_Lab__outer7__seed17__n6__inner2__value0.001 | 20 → 15 | 23 | +0.004716 |
| main__Formal_Lab__outer0__seed17__n18__inner0__value0.0001 | 36 → 23 | 31 | -0.011467 |
| main__Formal_Lab__outer0__seed17__n18__inner1__value0.0001 | 44 → 44 | 52 | -0.002474 |
| main__Formal_Lab__outer0__seed17__n18__inner2__value0.0001 | 29 → 31 | 39 | -0.002034 |

Best epoch beyond 50: 1/12.
Hit 100-epoch cap: 0/12.
Cumulative fit seconds: 81.92.

This is a rerun, not checkpoint continuation. Common-prefix score differences are recorded to distinguish stochastic rerun variation from evidence after epoch 50. Do not replace the original 8-condition calibration scores with these selectively repeated results.
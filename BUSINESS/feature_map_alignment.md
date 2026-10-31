# Feature-map alignment (`feature_map_alignment`) — QML v2

Protocol: [`POCs/qml_v2/feature_map_alignment.py`](../POCs/qml_v2/feature_map_alignment.py)

## Why

The TFM corpus validates **ZZ** empirically. The lineamiento is: advantage
appears when the **feature map** (ZZ or another) is aligned with the data
generator — not “ZZ is magic”.

This module runs a **cross table**:

| Generator (labels) | Kernel QSVM-ZZ | Kernel QSVM-PauliY | SVM-RBF |
|---|---|---|---|
| ZZ (`sign(<ZZ>)`) | matched | mismatched | classical |
| Pauli-Y (`sign(<YY>)`) | mismatched | matched | classical |

Pauli-Y uses `pauli_feature_map(..., paulis=["Y","YY"])` — a distinct
geometry from `ZZFeatureMap`.

## Run

```bash
cd POCs
PYTHONPATH=. python3 -m qml_v2.feature_map_alignment --quick
PYTHONPATH=. python3 -m qml_v2.feature_map_alignment --n-total 120 --n-repeats 5
```

Results: [`feature_map_alignment_v2_results.md`](feature_map_alignment_v2_results.md).

## Headline (n=120, 5 repeats)

| Generator | Matched kernel | Acc | Best classical / mismatched |
|---|---|---|---|
| **ZZ** | **QSVM-ZZ** | **0.828** | RBF / PauliY ≈ 0.478 |
| **Pauli-Y** | **QSVM-PauliY** | **0.550** | RBF 0.517; **QSVM-ZZ 0.406** (worst) |

Matched wins both rows. On Pauli-Y data the ZZ kernel is *worse* than RBF —
switching the map switches who benefits. Same protocol, different geometry.

Results: [`feature_map_alignment_v2_results.md`](feature_map_alignment_v2_results.md).

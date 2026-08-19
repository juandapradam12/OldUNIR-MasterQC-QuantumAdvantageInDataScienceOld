# Classification — code walkthrough (notebook 01)

Companion to [`POCs/notebooks/01_clasificacion_quantum_vs_clasico.ipynb`](../POCs/notebooks/01_clasificacion_quantum_vs_clasico.ipynb).

Open the notebook. We go cell by cell. This file is the shared notes from that pass.

Notebook 06 (kernel threshold) is the next file, not this one.

---

## What this notebook actually is

The opening markdown still describes a modular package (`DatasetLoader`, `src/models/quantum/`, …). **That package is not in the repo.** Everything is inlined in cell 2.

The notebook really runs **three experiments**:

1. **Telco Churn** (Kaggle) — sections 2–10. Quantum on a tiny PCA subset; classical on the full table. This is the *unfair* comparison you later corrected in §16.
2. **Quantum-native synthetic** (Liu-style) — sections 11–12, 15. Labels come from the same circuit as the kernel. This is where QSVM is *supposed* to win.
3. **SKG incidents** (practice data) — section 13. Same pattern as Telco: quantum on a subsample + PCA; classical on the full feature table.

Section 16 is the important methodological fix: **same SVC, same C, same 2-D PCA, only the kernel changes** (ZZ vs RBF), on all three datasets.

---

## Cell 2 — the two quantum models (this is the engine)

`N_QUBITS = 4` and `SAMPLE_SIZE = 500` are declared, then later ignored. Telco quantum actually uses **100 train / 50 test**. Quantum-native and §16 use **2 qubits**. Keep that mismatch in mind when you read results.

### Quantum Kernel SVM

```python
class QuantumKernelClassifier:
    def __init__(..., feature_dimension=4, reps=2, entanglement='full', ...):
        fm = ZZFeatureMap(feature_dimension=..., reps=reps, entanglement=entanglement)
        self.qkernel = FidelityQuantumKernel(feature_map=fm)
        self.model = None
    def fit(self, X, y):
        K = self.qkernel.evaluate(x_vec=X)
        self.model = SklearnSVC(kernel='precomputed', random_state=42)
        self.model.fit(K, y)
    def predict(self, X, X_train):
        K = self.qkernel.evaluate(x_vec=X, y_vec=X_train)
        return self.model.predict(K)
```

What this does, in order:

1. `ZZFeatureMap` encodes each row `x` as a circuit \(U_{\phi(x)}\) on `feature_dimension` qubits, repeated `reps` times, with `full` entanglement (every pair of qubits gets a ZZ interaction).
2. `FidelityQuantumKernel.evaluate` builds a Gram matrix. Entry \((i,j)\) is the **fidelity** \(|\langle \psi(x_i) | \psi(x_j) \rangle|^2\) — how much the two encoded states overlap. Train kernel is \(n \times n\); test kernel is \(n_{\text{test}} \times n_{\text{train}}\).
3. A **classical** `SVC(kernel='precomputed')` learns on that matrix. The “quantum” part is only the similarity; the decision is SVM.

Cost: \(O(n^2)\) circuit evaluations on train, plus \(n_{\text{test}} \cdot n_{\text{train}}\) at predict. That is why the Telco quantum subset is tiny.

`predict` needs `X_train` again because a precomputed kernel SVM cannot score a new point without the train Gram block. Easy to get wrong if you later extract a class.

### VQC

```python
class QuantumVQC:
    fm = ZZFeatureMap(...)
    ansatz = RealAmplitudes(num_qubits=feature_dimension, reps=reps, entanglement='linear')
    self.model = QiskitVQC(feature_map=fm, ansatz=ansatz, optimizer=COBYLA(maxiter=max_iter))
```

Different idea: a **trainable** circuit (RealAmplitudes) after the same feature map. COBYLA updates parameters to classify. No kernel matrix, but a harder optimization (barren plateaus, local minima). In this notebook VQC is a secondary baseline; the thesis story is the kernel.

Classical wrappers (`SVMClassifier`, `RFClassifier`, …) are thin sklearn skins. No hyperparameter search.

---

## Cells 4–9 — Telco data path

- Load `../data/input/telco_customer_churn.csv`, drop `customerID`, target `Churn` → 0/1.
- Label-encode categoricals independently per column (no `OneHotEncoder`). Fine for trees; slightly ugly for linear/SVM, but everything is scaled after.
- `TotalCharges` coerced to numeric; median fill.
- Stratified 80/20 split, then `StandardScaler`.
- Extra features: `charge_ratio = TotalCharges / (tenure * MonthlyCharges + 1)` and `num_services`. Then **re-scale**.

Nothing quantum yet. This is a standard churn table.

---

## Cell 11 — why Telco quantum cannot see the same problem

```python
pca = PCA(n_components=N_QUBITS)          # 4 components
X_train_q = pca.fit_transform(X_train_sc)
# then subsample 100 train / 50 test
mms = MinMaxScaler(feature_range=(-np.pi, np.pi))
```

Three compressions stacked:

| Step | Effect |
|---|---|
| PCA to 4 dims | Drops most of the table. Print `explained_variance_ratio_.sum()` when you re-run. |
| 100 / 50 rows | Classical models in cell 18 still get **all** train rows and **all** features. |
| Scale to \([-\pi, \pi]\) | Required so ZZFeatureMap angles are in a sensible range. |

Cell 18 then fits RF/XGBoost/GB on `X_train_sc` (full width, full n) and scores `X_test_sc`. Comparing that F1 to QSVM on 100×4 PCA is **not** a test of quantum vs classical. It is a test of “small kernel on a 4-D shadow” vs “gradient boosting on the real table.”

You already wrote this in cell 23. Section 16 is the fix. When we talk Telco numbers from sections 8–10, treat them as a **pipeline demo**, not as the scientific result.

Also: test indices are `range(min(50, n_test))` — the **first 50 test rows**, not a stratified draw. Train subsample is stratified; test is not. That can swing accuracy on a 50-point set.

---

## Cells 14–18 — Telco fit

- Cell 14: `QuantumKernelClassifier` on the 100/50 subset.
- Cell 16: VQC on the same subset, COBYLA 100 iterations.
- Cell 18: four classical models on the **full** scaled table; `class_weight` / `scale_pos_weight` for imbalance. Quantum models have **no** class-weight equivalent in this code.

So even the label imbalance is treated differently.

---

## Cells 24–25 — quantum-native (this is the positive control)

Labels are not “churn.” They are the sign of an observable on the encoded state:

```python
fm_adv = ZZFeatureMap(feature_dimension=2, reps=2, entanglement='full')
obs_zz = SparsePauliOp.from_list([("ZZ", 1.0)])
X_adv = rng.uniform(-np.pi, np.pi, (200, 2))
y_adv = 1 if Statevector(fm_adv.assign_parameters(...)).expectation_value(obs_zz).real >= 0 else 0
```

For each random \(x \in [-\pi,\pi]^2\):

1. Prepare \(|\psi(x)\rangle = U_{\mathrm{ZZ}}(x)|0\rangle\) (same family as the kernel).
2. Measure \(\langle ZZ \rangle\).
3. Label = sign of that number.

Then QSVM uses **the same** `ZZFeatureMap` as the kernel. Classical SVM-RBF / poly / linear / GB see only the 2-D points, not the circuit.

This is Liu 2021-style: advantage by **construction**. If QSVM did *not* win here, the kernel implementation would be suspect. If it *does* win, you have not shown that Telco or SKG should use QSVM. You have shown the kernel is working.

Same train/test split (30%) for quantum and classical — this cell is **fair**.

---

## Cell 28 — scalability on the synthetic set

Regenerates a **new** 300-point set (not the 200 from cell 25). Test is a 30% split, but train slices are `X_sc[:int(0.7*N)][:n_tr]` — a prefix of the array, **not** the complement of that test split. So train and test can overlap. For a published curve, fix this: one split, then nested prefixes of the *training* fold only.

Compares QSVM vs SVM-RBF only (not GB). That is the right pair if the claim is “kernel vs kernel.”

---

## Cells 29–30 — SKG incidents (practice data)

Hardcoded path:

```python
SKG_PATH = Path("/Users/juanprada/Documents/TFM_ComputacionCuantica/Datos/incidents_movility.csv")
```

In this repo the file is [`Datos/incidents_movility.csv`](../Datos/incidents_movility.csv). The notebook will not run here until that path is relative.

Task: gravity 25 vs 26 (simple vs injured). Features: `priority, type, zone_id, orient, implicated, lat, lon`.

Quantum: balanced subsample 180+180, PCA to `N_QUBITS` (**4**, despite the markdown saying 2), scale to \([-\pi,\pi]\), QSVM.

Classical: `train_test_split` on the **full** `X_skg` (all rows, 7 features, no PCA). Same unfairness as Telco.

Hypothesis in the markdown (“quantum captures geospatial nonlinearities”) is **not** what the code tests, because classical GB sees lat/lon and the other five columns on tens of thousands of rows, while QSVM sees 4 PCA axes on ~120 points.

---

## Cells 36–37 — the fair kernel-vs-kernel test

`evaluate_pair`:

- Cap train at 120.
- PCA to **2** components (not 4).
- MinMax to \([-\pi,\pi]\).
- QSVM: fidelity kernel + `SVC(kernel='precomputed', C=1)`.
- SVM-RBF: `SVC(kernel='rbf', C=1)` on the **same** 2-D scaled points.

This isolates “ZZ fidelity vs RBF” and is the comparison that belongs in a paper or a client conversation.

Caveats still in this function:

- PCA is applied even to quantum-native data that is **already** 2-D in \([-\pi,\pi]\). Harmless-ish, but not identity.
- SKG here uses only `priority, implicated, latitude, longitude` (4 cols), not the 7 from §13. Two different SKG experiments.
- Single split, no k-fold. Deltas on n≈80 test points are noisy.
- Telco still compressed to 2 PCA axes (~30% variance in the thesis). Fair *between kernels*, unfair vs a GB that could use 20 features — but §16 does not claim vs GB.

---

## Open issues to fix after we finish reading (not yet)

1. Relative `Datos/` path; `N_QUBITS` 4 vs 2 used inconsistently.
2. Telco/SKG headline tables: always pair them with §16 or a matched-n classical run.
3. Stratify the Telco quantum test subsample.
4. Scalability: disjoint train/test.
5. Extract `QuantumKernelClassifier` into `POCs/src/` once we agree the API (`predict` needs `X_train`).

---

## Next

Notebook **06** (threshold / sample size on Telco and SKG under the kernel). Then we decide what to change in 01.

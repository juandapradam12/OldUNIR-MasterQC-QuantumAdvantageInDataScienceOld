# Time-series forecasting — fair lag protocol

Protocol: [`POCs/qml_v2/timeseries.py`](../POCs/qml_v2/timeseries.py)

## Idea

One-step forecast from a lag window. **Not** an ARIMA bake-off.

| Table | What each model sees |
|---|---|
| A | Same `n_lags` (= `n_qubits`) features → QK-SVR-ZZ vs Ridge / SVR-RBF / GB |
| B | Classical gets extra lags, lag mean/std and (SKG) cyclic hour (matched n) |

Splits are **time-ordered** (70 % past for training, 30 % future for test). Two naive
references are reported on the same test windows: persistence (`y_{t-1}`) and the training mean.

## Datasets

| Dataset | Construction | Expectation |
|---|---|---|
| `zz_lag` | Independent lag pairs in ZZ angle space; target = ⟨ZZ⟩ (native control, not a dynamic series) | QK-SVR may win |
| `sine` | Periodic classical series + noise | classical |
| `skg_speed` | Speed of real SKG corridors (5-min readings, 144 per corridor); one corridor per repeat | classical |

## Run

```bash
cd POCs
python3 -m qml_v2.timeseries --quick
python3 -m qml_v2.timeseries --n-total 160 --n-repeats 5
```

Results: [`timeseries_v2_results.md`](timeseries_v2_results.md).

## Headline (5 repeats, time-ordered; MAE on test)

| Dataset | Table A winner | QK-SVR-ZZ | Naive references |
|---------|----------------|-----------|------------------|
| **zz_lag** | **QK-SVR 0.056** (Ridge 0.414) | 0.056 | train mean 0.404 |
| sine | SVR-RBF 0.260 | 0.662 | train mean 0.658, persistence 0.287 |
| skg_speed | Ridge 2.80 | 13.59 | train mean 13.61, persistence 2.75 |

Same story as classification/regression: quantum wins only on ZZ-structured
targets. Outside that regime QK-SVR matches the training-mean predictor, and Ridge
sits at the level of persistence. Figures and table: notebook 12.

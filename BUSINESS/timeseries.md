# Time-series forecasting — fair lag protocol

Protocol: [`POCs/qml_v2/timeseries.py`](../POCs/qml_v2/timeseries.py)

## Idea

One-step forecast from a lag window. **Not** an ARIMA bake-off.

| Table | What each model sees |
|---|---|
| A | Same `n_lags` (= `n_qubits`) features → QK-SVR-ZZ vs Ridge / SVR-RBF / GB |
| B | Classical gets extra lags + hour cyclic features (matched n) |

Splits are **time-ordered** (train = past, test = future).

## Datasets

| Dataset | Construction | Expectation |
|---|---|---|
| `zz_lag` | Lag window in ZZ angle space; target = ⟨ZZ⟩ | QK-SVR may win |
| `sine` | Periodic classical series + noise | classical |
| `skg_speed` | One corridor speed series (practice) | classical Table B |

## Run

```bash
cd POCs
python3 -m qml_v2.timeseries --quick
python3 -m qml_v2.timeseries --n-total 160 --n-repeats 5
```

Results: [`timeseries_v2_results.md`](timeseries_v2_results.md).

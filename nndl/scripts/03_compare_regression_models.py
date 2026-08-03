"""
Compares candidate architectures for the 24h-ahead AQI regression forecaster and
picks the best one by test-set RMSE. Produced because the project was simplified
to use a single model (the regression forecaster) instead of a regression model
plus a separate attention-augmented classifier -- this script is the empirical
justification for which regression architecture that single model should be.

All candidates share: same 48h lookback / 24h horizon windows, same chronological
splits (train <=2022, val 2023, test 2024), same feature scaling (mu/sd computed
on train only), same optimizer/loss/early-stopping protocol. Only the architecture
differs, so any RMSE difference is attributable to architecture choice.

Run: python3 03_compare_regression_models.py
Outputs: ../model/comparison_metrics.json, ../model/comparison_histories.json
"""
import json
import time
import numpy as np
import pandas as pd
import keras
from keras import layers
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOOKBACK, HORIZON = 48, 24
np.random.seed(42)
keras.utils.set_random_seed(42)

# ---------------------------------------------------------------------------
# Data (identical prep to 02_aqi_lstm.ipynb)
# ---------------------------------------------------------------------------
df = pd.read_csv(ROOT / "data" / "hebbal_clean.csv", parse_dates=["datetime"], index_col="datetime")
df['wd_sin'] = np.sin(np.deg2rad(df.wind_dir)); df['wd_cos'] = np.cos(np.deg2rad(df.wind_dir))
df['hour_sin'] = np.sin(2*np.pi*df.index.hour/24); df['hour_cos'] = np.cos(2*np.pi*df.index.hour/24)
df['doy_sin'] = np.sin(2*np.pi*df.index.dayofyear/365); df['doy_cos'] = np.cos(2*np.pi*df.index.dayofyear/365)
df['dow'] = df.index.dayofweek / 6.0
FEATURES = ['AQI', 'temp', 'rh', 'wind_speed', 'precip', 'pressure',
            'wd_sin', 'wd_cos', 'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos', 'dow']

X_raw = df[FEATURES].to_numpy(np.float32)
y_raw = df['AQI'].to_numpy(np.float32)
n = len(df)
ok = ~np.isnan(X_raw).any(axis=1)
yok = ~np.isnan(y_raw)
idx = np.array([t for t in range(LOOKBACK, n - HORIZON)
                if ok[t-LOOKBACK:t].all() and yok[t:t+HORIZON].all()])
times = df.index
train_m = times[idx] <= '2022-12-31 23:00'
val_m = (times[idx] > '2022-12-31 23:00') & (times[idx] <= '2023-12-31 23:00')
test_m = times[idx] > '2023-12-31 23:00'

tr_span = X_raw[: idx[train_m].max()]
mu = np.nanmean(tr_span, axis=0); sd = np.nanstd(tr_span, axis=0); sd[sd == 0] = 1
y_mu = np.nanmean(y_raw[: idx[train_m].max()]); y_sd = np.nanstd(y_raw[: idx[train_m].max()])

Xn = (X_raw - mu) / sd
X = np.stack([Xn[t-LOOKBACK:t] for t in idx]).astype(np.float32)
Y = np.stack([y_raw[t:t+HORIZON] for t in idx]).astype(np.float32)
Yn = (Y - y_mu) / y_sd

Xtr, Ytr = X[train_m], Yn[train_m]
Xva, Yva = X[val_m], Yn[val_m]
Xte, Yte_raw = X[test_m], Y[test_m]
print('train/val/test:', Xtr.shape, Xva.shape, Xte.shape, flush=True)

n_features = len(FEATURES)

# ---------------------------------------------------------------------------
# Baselines (no training)
# ---------------------------------------------------------------------------
def report(name, p, true):
    rmse = float(np.sqrt(np.mean((p - true) ** 2)))
    mae = float(np.mean(np.abs(p - true)))
    hs = {h: float(np.sqrt(np.mean((p[:, h] - true[:, h]) ** 2))) for h in [0, 5, 11, 23]}
    print(f'{name:16s} RMSE {rmse:6.2f}  MAE {mae:6.2f} | '
          f'RMSE@1h {hs[0]:.1f}  @6h {hs[5]:.1f}  @12h {hs[11]:.1f}  @24h {hs[23]:.1f}', flush=True)
    return {'rmse': rmse, 'mae': mae, 'rmse_by_h': {str(k+1): v for k, v in hs.items()}}

results = {}
last_obs = Xte[:, -1, 0] * sd[0] + mu[0]
persist = np.repeat(last_obs[:, None], HORIZON, axis=1)
seas = Xte[:, -24:, 0] * sd[0] + mu[0]
results['Persistence'] = report('Persistence', persist, Yte_raw)
results['SeasonalNaive'] = report('SeasonalNaive', seas, Yte_raw)

# ---------------------------------------------------------------------------
# Candidate architectures
# ---------------------------------------------------------------------------
def build_stacked_lstm():
    return keras.Sequential([
        layers.Input((LOOKBACK, n_features)),
        layers.LSTM(64, return_sequences=True, dropout=0.2),
        layers.LSTM(32, dropout=0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(HORIZON),
    ], name="StackedLSTM")

def build_single_lstm():
    return keras.Sequential([
        layers.Input((LOOKBACK, n_features)),
        layers.LSTM(64, dropout=0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(HORIZON),
    ], name="SingleLSTM")

def build_stacked_gru():
    return keras.Sequential([
        layers.Input((LOOKBACK, n_features)),
        layers.GRU(64, return_sequences=True, dropout=0.2),
        layers.GRU(32, dropout=0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(HORIZON),
    ], name="StackedGRU")

def build_bilstm():
    return keras.Sequential([
        layers.Input((LOOKBACK, n_features)),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2)),
        layers.LSTM(32, dropout=0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(HORIZON),
    ], name="BiLSTM")

def build_attention_lstm():
    inp = layers.Input((LOOKBACK, n_features))
    h1 = layers.LSTM(64, return_sequences=True, dropout=0.2)(inp)
    h2 = layers.LSTM(32, return_sequences=True, dropout=0.2)(h1)
    score = layers.Dense(1)(h2)
    weights = layers.Softmax(axis=1)(score)
    context = layers.Multiply()([h2, weights])
    pooled = layers.Lambda(lambda t: keras.ops.sum(t, axis=1))(context)
    d1 = layers.Dense(64, activation='relu')(pooled)
    out = layers.Dense(HORIZON)(d1)
    return keras.Model(inp, out, name="AttentionLSTM")

CANDIDATES = {
    "StackedLSTM": build_stacked_lstm,
    "SingleLSTM": build_single_lstm,
    "StackedGRU": build_stacked_gru,
    "BiLSTM": build_bilstm,
    "AttentionLSTM": build_attention_lstm,
}

MAX_EPOCHS = 40
PATIENCE = 6
histories = {}
for name, builder in CANDIDATES.items():
    print(f"\n=== training {name} ===", flush=True)
    t0 = time.time()
    keras.utils.set_random_seed(42)
    model = builder()
    n_params = model.count_params()
    model.compile(keras.optimizers.Adam(5e-4), 'mse', metrics=['mae'])
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    hist = model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=MAX_EPOCHS,
                      batch_size=512, callbacks=[es], verbose=2)
    pred = model.predict(Xte, batch_size=1024, verbose=0) * y_sd + y_mu
    r = report(name, pred, Yte_raw)
    r['n_params'] = int(n_params)
    r['epochs_trained'] = len(hist.history['loss'])
    r['train_seconds'] = round(time.time() - t0, 1)
    results[name] = r
    histories[name] = {'loss': hist.history['loss'], 'val_loss': hist.history['val_loss']}
    if name == "StackedLSTM":
        model.save(ROOT / "model" / "lstm_aqi_refit.keras")
    print(f"{name}: {r['n_params']} params, {r['epochs_trained']} epochs, "
          f"{r['train_seconds']}s", flush=True)
    json.dump(results, open(ROOT / "model" / "comparison_metrics.json", "w"), indent=1)
    json.dump(histories, open(ROOT / "model" / "comparison_histories.json", "w"), indent=1)

best = min((k for k in results if k not in ("Persistence", "SeasonalNaive")),
           key=lambda k: results[k]['rmse'])
print(f"\n=== BEST MODEL: {best} (RMSE {results[best]['rmse']:.2f}) ===", flush=True)
results['_best'] = best
json.dump(results, open(ROOT / "model" / "comparison_metrics.json", "w"), indent=1)
print("done", flush=True)

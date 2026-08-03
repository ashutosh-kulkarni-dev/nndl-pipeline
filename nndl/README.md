# NNDL - AQI Forecasting (Hebbal, Bengaluru) with LSTM

## Folder layout

```
nndl/
  scripts/     01_clean_data.py, 02_aqi_lstm.ipynb, 03_compare_regression_models.py
  data/
    raw/       the three original raw source files
    hebbal_clean.csv   (produced by 01_clean_data.py)
  model/       lstm_aqi.keras, metrics.json  (produced by 02_aqi_lstm.ipynb)
  figures/     fig1_series.png ... fig7_scatter.png  (produced by 02_aqi_lstm.ipynb)
```

`model/lstm_aqi.keras` and `model/metrics.json` already exist from a prior run, so you
only need to redo the steps below if you want to reproduce them from scratch or run the
architecture comparison.

## 1. Install dependencies

```bash
cd nndl
pip install -r requirements.txt --break-system-packages
```

(Drop `--break-system-packages` if you're using a virtual environment, which is
recommended: `python3 -m venv venv && source venv/bin/activate` first.)

TensorFlow is a large download (several hundred MB) - on a slow connection the install
can take a few minutes.

## 2. Rebuild the cleaned dataset (optional - already present)

```bash
cd nndl/scripts
python3 01_clean_data.py
```

Reads the three files in `../data/raw/`, writes `../data/hebbal_clean.csv`.

## 3. Train the baseline LSTM and generate figures

```bash
cd nndl/scripts
jupyter nbconvert --to notebook --execute --inplace 02_aqi_lstm.ipynb
```

Or open `02_aqi_lstm.ipynb` in Jupyter/VS Code and run all cells. This writes
`../model/lstm_aqi.keras`, `../model/metrics.json`, and `../figures/fig1..7.png`.

## 4. Run the architecture comparison

```bash
cd nndl/scripts
python3 03_compare_regression_models.py
```

Trains five candidate architectures (StackedLSTM, SingleLSTM, StackedGRU, BiLSTM,
AttentionLSTM) under an identical protocol and writes:
- `../model/comparison_metrics.json` - RMSE/MAE per architecture, plus `_best`
- `../model/comparison_histories.json` - training/validation loss curves per architecture
- `../model/lstm_aqi_refit.keras` - the refitted StackedLSTM from this run

This trains 5 models end-to-end (up to 40 epochs each with early stopping); expect this
to take a while on CPU. Once it finishes, check `comparison_metrics.json` for the
`_best` field and the per-model RMSE to see which architecture actually won - plug those
numbers into Section 3.5.2 / 4.2 of the report.

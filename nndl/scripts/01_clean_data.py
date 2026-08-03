"""
Step 1 - Data cleaning & unification for Hebbal (Bengaluru) AQI forecasting.

Inputs (expected under ../data/raw relative to this script):
  1. AQI hebbal 2023.csv            - KSPCB hourly AQI, pivoted layout (2017-2023;
                                      2017 to mid-2018 are empty headers)
  2. blr-hebbal-kspcb-2024-25.csv   - KSPCB 15-min raw pollutant concentrations (2024-25)
  3. hebbal_weather_2017_2024.csv   - Open-Meteo ERA5 hourly weather, IST

Output:
  ../data/hebbal_clean.csv - one row per hour, Jul-2018 .. Dec-2024:
      datetime, AQI, temp, rh, wind_speed, wind_dir, precip, pressure

Notes on decisions (explained in the report):
  * KSPCB timestamps carry a fake "+0000" suffix but are actually IST
    (verified: temperature peaks at 14:00 as-labelled). We strip the timezone.
  * AQI for 2024 is computed from raw concentrations with CPCB breakpoints:
    sub-index per pollutant via breakpoint interpolation, AQI = max sub-index,
    valid only if >=3 sub-indices exist incl. one of PM2.5/PM10.
    Sub-indices use HOURLY mean concentrations (not 24-h rolling means):
    the official historical hourly AQI series changes ~8-11 points/hour,
    which is only reproducible from hourly concentrations. Using rolling
    means would make 2024 ~10x smoother than 2018-23 and bias the model.
  * 2025 pollutant data is dropped because weather covers only up to Dec-2024.
  * Gaps <= 6 h are linearly interpolated; longer gaps stay NaN and are later
    excluded when building training windows (never imputed blindly).
"""
import csv
import calendar
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

AQI_PIVOT_FILE = ROOT / "data" / "raw" / "AQI hebbal 2023.csv"
POLLUTANT_FILE = ROOT / "data" / "raw" / "blr-hebbal-kspcb-2024-25.csv"
WEATHER_FILE = ROOT / "data" / "raw" / "hebbal_weather_2017_2024.csv"
OUTPUT_FILE = ROOT / "data" / "hebbal_clean.csv"

MAX_GAP_INTERPOLATE = 6  # hours


# --------------------------------------------------------------------------
# 1. Parse the pivoted historical AQI file (2017-2023)
#    Layout: "Year,YYYY" marker rows; "Month-YYYY,00:00,...,23:00" header rows;
#    then one row per day-of-month with 24 hourly AQI values.
# --------------------------------------------------------------------------
def parse_pivoted_aqi(path: Path) -> pd.DataFrame:
    months = {m: i for i, m in enumerate(calendar.month_name) if m}
    records = []
    current = None  # (year, month)

    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip() or row[0].strip() == '"':
                continue
            head = row[0].strip()
            if head == "Year":
                continue
            if "-" in head and head.split("-")[0] in months:  # month header
                mon_name, year = head.split("-")
                current = (int(year), months[mon_name])
                continue
            if head.isdigit() and current:  # day row
                year, month = current
                day = int(head)
                if day > calendar.monthrange(year, month)[1]:
                    continue  # guard against stray rows
                for hour, cell in enumerate(row[1:25]):
                    cell = cell.strip()
                    if cell:
                        try:
                            records.append(
                                (pd.Timestamp(year, month, day, hour), float(cell))
                            )
                        except ValueError:
                            pass  # non-numeric artefacts

    df = (
        pd.DataFrame(records, columns=["datetime", "AQI"])
        .set_index("datetime")
        .sort_index()
    )
    return df[~df.index.duplicated(keep="first")]


# --------------------------------------------------------------------------
# 2. Compute CPCB AQI from raw pollutant concentrations (2024)
# --------------------------------------------------------------------------
# CPCB breakpoints: (conc_low, conc_high, aqi_low, aqi_high)
BREAKPOINTS = {
    "PM2.5": [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200),
              (90, 120, 201, 300), (120, 250, 301, 400), (250, 500, 401, 500)],
    "PM10":  [(0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200),
              (250, 350, 201, 300), (350, 430, 301, 400), (430, 600, 401, 500)],
    "NO2":   [(0, 40, 0, 50), (40, 80, 51, 100), (80, 180, 101, 200),
              (180, 280, 201, 300), (280, 400, 301, 400), (400, 600, 401, 500)],
    "SO2":   [(0, 40, 0, 50), (40, 80, 51, 100), (80, 380, 101, 200),
              (380, 800, 201, 300), (800, 1600, 301, 400), (1600, 2400, 401, 500)],
    "NH3":   [(0, 200, 0, 50), (200, 400, 51, 100), (400, 800, 101, 200),
              (800, 1200, 201, 300), (1200, 1800, 301, 400), (1800, 2400, 401, 500)],
    "CO":    [(0, 1, 0, 50), (1, 2, 51, 100), (2, 10, 101, 200),
              (10, 17, 201, 300), (17, 34, 301, 400), (34, 50, 401, 500)],
    "O3":    [(0, 50, 0, 50), (50, 100, 51, 100), (100, 168, 101, 200),
              (168, 208, 201, 300), (208, 748, 301, 400), (748, 1000, 401, 500)],
}


def sub_index(conc: pd.Series, pollutant: str) -> pd.Series:
    """Linear interpolation of concentration -> AQI sub-index within CPCB bands."""
    si = pd.Series(np.nan, index=conc.index)
    for lo, hi, aqi_lo, aqi_hi in BREAKPOINTS[pollutant]:
        mask = (conc >= lo) & (conc <= hi)
        si[mask] = aqi_lo + (conc[mask] - lo) * (aqi_hi - aqi_lo) / (hi - lo)
    si[conc > BREAKPOINTS[pollutant][-1][1]] = 500.0
    return si


def compute_cpcb_aqi(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, parse_dates=["Timestamp"])
    raw["Timestamp"] = raw["Timestamp"].dt.tz_localize(None)  # labels are IST
    raw = raw.set_index("Timestamp").sort_index()

    col_map = {
        "PM2.5 (µg/m³)": "PM2.5", "PM10 (µg/m³)": "PM10", "NO2 (µg/m³)": "NO2",
        "SO2 (µg/m³)": "SO2", "NH3 (µg/m³)": "NH3", "CO (mg/m³)": "CO",
        "Ozone (µg/m³)": "O3",
    }
    pol = (
        raw[list(col_map)]
        .rename(columns=col_map)
        .apply(pd.to_numeric, errors="coerce")
        .resample("h")
        .mean()
    )

    si = pd.DataFrame({p: sub_index(pol[p], p) for p in BREAKPOINTS})
    enough = si.notna().sum(axis=1) >= 3
    has_pm = si[["PM2.5", "PM10"]].notna().any(axis=1)
    aqi = si.max(axis=1).where(enough & has_pm).rename("AQI").to_frame().dropna()
    return aqi


# --------------------------------------------------------------------------
# 3. Weather
# --------------------------------------------------------------------------
def load_weather(path: Path) -> pd.DataFrame:
    w = pd.read_csv(path, parse_dates=["time"]).set_index("time")
    w.columns = ["temp", "rh", "wind_speed", "wind_dir", "precip", "pressure"]
    return w


# --------------------------------------------------------------------------
# 4. Merge + clean
# --------------------------------------------------------------------------
def main() -> None:
    aqi_hist = parse_pivoted_aqi(AQI_PIVOT_FILE)
    print(f"[1/4] Historical AQI : {len(aqi_hist):6d} h  "
          f"({aqi_hist.index.min()} -> {aqi_hist.index.max()})")

    aqi_2024 = compute_cpcb_aqi(POLLUTANT_FILE)
    print(f"[2/4] Computed AQI   : {len(aqi_2024):6d} h  "
          f"({aqi_2024.index.min()} -> {aqi_2024.index.max()})")
    print(f"      sanity: hist mean {aqi_hist.AQI.mean():.1f} / "
          f"computed mean {aqi_2024.AQI.mean():.1f} (should be similar)")

    weather = load_weather(WEATHER_FILE)
    print(f"[3/4] Weather        : {len(weather):6d} h")

    aqi_all = pd.concat([aqi_hist, aqi_2024]).sort_index()
    aqi_all = aqi_all[~aqi_all.index.duplicated(keep="first")]

    # full hourly grid; start where real AQI data begins, end where weather ends
    start = max(aqi_all.index.min(), weather.index.min())
    end = weather.index.max()
    grid = pd.date_range(start, end, freq="h")
    df = pd.DataFrame(index=grid).join(aqi_all).join(weather)
    df.index.name = "datetime"

    missing_before = df.AQI.isna().mean()
    df["AQI"] = df.AQI.interpolate(limit=MAX_GAP_INTERPOLATE, limit_area="inside")
    df["AQI"] = df.AQI.clip(0, 500)
    for c in ["temp", "rh", "wind_speed", "wind_dir", "precip", "pressure"]:
        df[c] = df[c].interpolate(limit=MAX_GAP_INTERPOLATE, limit_area="inside")

    print(f"[4/4] Unified        : {len(df):6d} h  "
          f"AQI missing {missing_before:.1%} -> {df.AQI.isna().mean():.1%} "
          f"(gaps <= {MAX_GAP_INTERPOLATE} h interpolated, longer gaps kept as NaN)")

    df.to_csv(OUTPUT_FILE)
    print(f"\nSaved -> {OUTPUT_FILE.name}   shape={df.shape}")


if __name__ == "__main__":
    main()

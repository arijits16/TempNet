"""
Step 1: Generate a SYNTHETIC daily weather dataset and save it as a CSV file.

NOTE: This is synthetic demonstration data, NOT real weather observations.
Every value is produced locally with NumPy using seasonal curves,
simple physical-style relationships and random noise.

Run from the project root:
    python -m src.generate_dataset
"""

import numpy as np
import pandas as pd

from src import DATA_DIR, RAW_CSV, SEED

N_DAYS = 500
START_DATE = "2025-01-01"


def generate_weather(n_days: int = N_DAYS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=n_days, freq="D")
    day_of_year = dates.day_of_year.to_numpy()

    # Smooth seasonal curve: -1 in mid-winter, +1 around late June (summer)
    season = np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    # A second curve that peaks in mid-August: used for a cloudy/rainy season
    wet_season = np.sin(2 * np.pi * (day_of_year - 135) / 365.25)

    cloud = np.zeros(n_days)
    humidity = np.zeros(n_days)
    rainfall = np.zeros(n_days)
    wind = np.zeros(n_days)
    solar = np.zeros(n_days)
    t_max = np.zeros(n_days)
    t_min = np.zeros(n_days)

    for t in range(n_days):
        # Cloud cover (%): persists from day to day and rises in the wet season
        cloud_base = 45 + 20 * wet_season[t]
        prev_cloud = cloud[t - 1] if t > 0 else cloud_base
        cloud[t] = np.clip(0.6 * prev_cloud + 0.4 * cloud_base + rng.normal(0, 14), 0, 100)

        # Rainfall (mm): only on some days, more likely when it is cloudy
        rain_prob = 1 / (1 + np.exp(-(cloud[t] - 65) / 7))
        if rng.random() < rain_prob:
            rainfall[t] = rng.gamma(shape=1.8, scale=3 + 0.08 * cloud[t])

        # Humidity (%): higher with clouds and rain
        humidity[t] = np.clip(
            38 + 0.35 * cloud[t] + 1.2 * min(rainfall[t], 15) + 5 * wet_season[t] + rng.normal(0, 5),
            5, 100,
        )

        # Wind speed (km/h): a gamma-distributed breeze, slightly stronger on rainy days
        wind[t] = rng.gamma(shape=4, scale=3) + 0.3 * min(rainfall[t], 20)

        # Solar radiation (W/m^2): clear-sky value from the season, reduced by clouds
        clear_sky = 230 + 70 * season[t]
        solar[t] = max(0.0, clear_sky * (1 - 0.6 * cloud[t] / 100) + rng.normal(0, 12))

        # Maximum temperature (deg C): seasonal level + yesterday's anomaly
        # + sunshine warms the day, clouds and rain cool it
        base_max = 29 + 7 * season[t]
        prev_anom_max = (t_max[t - 1] - (29 + 7 * season[t - 1])) if t > 0 else 0.0
        t_max[t] = (
            base_max
            + 0.65 * prev_anom_max
            + 0.025 * (solar[t] - 0.75 * clear_sky)
            - 0.02 * (cloud[t] - 45)
            - 0.08 * min(rainfall[t], 20)
            + rng.normal(0, 0.9)
        )

        # Minimum temperature (deg C): seasonal level + recent temperatures + noise
        # Cloudy nights keep a little more heat, so cloud adds a small warming term
        base_min = 18 + 6 * season[t]
        prev_anom_min = (t_min[t - 1] - (18 + 6 * season[t - 1])) if t > 0 else 0.0
        t_min[t] = (
            base_min
            + 0.5 * prev_anom_min
            + 0.25 * (t_max[t] - base_max)
            + 0.015 * (cloud[t] - 45)
            + rng.normal(0, 0.8)
        )
        # Physical constraint: the minimum must stay below the maximum
        t_min[t] = min(t_min[t], t_max[t] - 2.0 - abs(rng.normal(0, 0.5)))

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "max_temperature": np.round(t_max, 1),
        "min_temperature": np.round(t_min, 1),
        "humidity": np.round(humidity, 1),
        "wind_speed": np.round(wind, 1),
        "rainfall": np.round(rainfall, 1),
        "cloud_cover": np.round(cloud, 1),
        "solar_radiation": np.round(solar, 1),
        "day": np.arange(1, n_days + 1),
    })
    return df


def validate(df: pd.DataFrame) -> None:
    """Fail loudly if any generated value breaks a dataset rule."""
    assert len(df.columns) == 9, "CSV must have exactly 9 columns"
    assert not df.isna().any().any(), "Missing values found"
    assert df["date"].is_unique, "Duplicate dates found"
    gaps = pd.to_datetime(df["date"]).diff().dropna()
    assert (gaps == pd.Timedelta(days=1)).all(), "Dates are not consecutive"
    assert (df["max_temperature"] > df["min_temperature"]).all(), "max <= min found"
    assert df["humidity"].between(0, 100).all(), "Humidity outside 0-100"
    assert df["cloud_cover"].between(0, 100).all(), "Cloud cover outside 0-100"
    for col in ["wind_speed", "rainfall", "solar_radiation"]:
        assert (df[col] >= 0).all(), f"Negative values in {col}"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_weather()
    validate(df)
    df.to_csv(RAW_CSV, index=False)

    print("SYNTHETIC demonstration data (not real weather observations)")
    print(f"Saved {len(df)} rows x {len(df.columns)} columns to {RAW_CSV}")
    print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print(f"Rainy days: {(df['rainfall'] > 0).sum()}")
    print(df.head())


if __name__ == "__main__":
    main()

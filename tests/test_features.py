import numpy as np
import pandas as pd

from seoul_bike_mlops.features import SEASONS, engineer_features


def _raw(n=48, season="Winter", functioning="Yes"):
    return pd.DataFrame(
        {
            "date": pd.date_range("2018-01-01", periods=n, freq="h").strftime("%d/%m/%Y"),
            "rented_bike_count": np.arange(n),
            "hour": np.arange(n) % 24,
            "temperature": np.linspace(-5, 5, n),
            "humidity": np.full(n, 40),
            "wind_speed": np.full(n, 1.2),
            "visibility": np.full(n, 2000),
            "dew_point_temperature": np.linspace(-8, 2, n),
            "solar_radiation": np.zeros(n),
            "rainfall": np.zeros(n),
            "snowfall": np.zeros(n),
            "season": season,
            "holiday": "No Holiday",
            "functioning_day": functioning,
        }
    )


def test_non_functioning_rows_are_dropped():
    df = _raw(24, functioning="No")
    assert len(engineer_features(df)) == 0


def test_cyclical_columns_are_bounded():
    feats = engineer_features(_raw())
    for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
        assert col in feats
        assert feats[col].between(-1, 1).all()


def test_season_columns_present_even_for_one_season():
    # only winter appears, but all four season columns must exist so the
    # reference and current feature tables line up
    feats = engineer_features(_raw(season="Winter"))
    for s in SEASONS:
        assert f"season_{s}" in feats
    assert (feats["season_Winter"] == 1).all()
    assert (feats["season_Summer"] == 0).all()


def test_dropped_raw_columns_gone():
    feats = engineer_features(_raw())
    for col in ["date", "holiday", "functioning_day", "dew_point_temperature"]:
        assert col not in feats

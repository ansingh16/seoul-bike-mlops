import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from seoul_bike_mlops.validate import SCHEMA


def _good_row():
    return {
        "date": "2018-01-01",
        "rented_bike_count": 254,
        "hour": 0,
        "temperature": -5.2,
        "humidity": 37,
        "wind_speed": 2.2,
        "visibility": 2000,
        "dew_point_temperature": -17.6,
        "solar_radiation": 0.0,
        "rainfall": 0.0,
        "snowfall": 0.0,
        "season": "Winter",
        "holiday": "No Holiday",
        "functioning_day": "Yes",
    }


def test_valid_frame_passes():
    df = pd.DataFrame([_good_row()])
    df["date"] = pd.to_datetime(df["date"])
    out = SCHEMA.validate(df, lazy=True)
    assert len(out) == 1


def test_hour_out_of_range_fails():
    bad = _good_row()
    bad["hour"] = 30
    df = pd.DataFrame([bad])
    df["date"] = pd.to_datetime(df["date"])
    with pytest.raises(SchemaErrors):
        SCHEMA.validate(df, lazy=True)


def test_unknown_season_fails():
    bad = _good_row()
    bad["season"] = "Monsoon"
    df = pd.DataFrame([bad])
    df["date"] = pd.to_datetime(df["date"])
    with pytest.raises(SchemaErrors):
        SCHEMA.validate(df, lazy=True)

"""End-to-end check of the serving API against a tiny in-memory model.

The real champion export is gitignored and absent in CI, so instead of loading it
we train a throwaway LightGBM on a handful of engineered rows and drop it straight
into the app's STATE. That still exercises the request schema, the feature
engineering path, and the log-target inversion.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from lightgbm import LGBMRegressor

from seoul_bike_mlops import serve
from seoul_bike_mlops.features import engineer_features


def _raw(n=200):
    rng = np.random.default_rng(0)
    hour = rng.integers(0, 24, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2018-01-01", periods=n, freq="h").strftime("%d/%m/%Y"),
            "rented_bike_count": (hour * 30 + rng.integers(0, 100, n)).astype(int),
            "hour": hour,
            "temperature": rng.uniform(-10, 30, n),
            "humidity": rng.integers(10, 90, n),
            "wind_speed": rng.uniform(0, 5, n),
            "visibility": rng.integers(200, 2000, n),
            "dew_point_temperature": rng.uniform(-15, 20, n),
            "solar_radiation": rng.uniform(0, 3, n),
            "rainfall": np.zeros(n),
            "snowfall": np.zeros(n),
            "season": rng.choice(["Winter", "Summer"], n),
            "holiday": "No Holiday",
            "functioning_day": "Yes",
        }
    )


@pytest.fixture
def client(monkeypatch):
    feats = engineer_features(_raw())
    y = feats.pop("rented_bike_count")
    model = LGBMRegressor(n_estimators=20, num_leaves=8, min_child_samples=5)
    model.fit(feats, np.log1p(y))

    # don't hit the registry / model dir on startup; inject the model ourselves
    monkeypatch.setattr(serve, "load_champion", lambda: None)
    serve.STATE["model"] = model
    serve.STATE["features"] = model.booster_.feature_name()
    serve.STATE["info"] = {"name": "test", "version": 0, "log_target": True}

    with TestClient(serve.app) as c:
        yield c

    serve.STATE.update({"model": None, "info": {}, "features": []})


def _obs(**over):
    base = {
        "date": "01/12/2018",
        "hour": 18,
        "temperature": -2.0,
        "humidity": 40,
        "wind_speed": 1.5,
        "visibility": 2000,
        "dew_point_temperature": -12.0,
        "solar_radiation": 0.0,
        "rainfall": 0.0,
        "snowfall": 0.0,
        "season": "Winter",
    }
    base.update(over)
    return base


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_nonnegative_count(client):
    r = client.post("/predict", json=_obs())
    assert r.status_code == 200
    pred = r.json()["predicted_rented_bike_count"]
    assert pred >= 0.0


def test_closed_day_predicts_zero(client):
    r = client.post("/predict", json=_obs(functioning_day="No"))
    assert r.status_code == 200
    assert r.json()["predicted_rented_bike_count"] == 0.0


def test_bad_hour_is_rejected(client):
    r = client.post("/predict", json=_obs(hour=42))
    assert r.status_code == 422

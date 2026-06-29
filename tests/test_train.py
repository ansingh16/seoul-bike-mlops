import numpy as np
import pandas as pd

from seoul_bike_mlops.train import evaluate, time_split


def test_time_split_keeps_order_and_fraction():
    df = pd.DataFrame({"x": range(100)})
    train, hold = time_split(df, holdout_fraction=0.2)
    assert len(train) == 80
    assert len(hold) == 20
    # the holdout is the tail in time order, never shuffled
    assert train["x"].iloc[-1] == 79
    assert hold["x"].iloc[0] == 80


def test_evaluate_reports_expected_keys():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    metrics = evaluate(y, y)
    assert set(metrics) == {"mae", "rmse", "r2"}
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_evaluate_penalises_error():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    pred = y + 5
    metrics = evaluate(y, pred)
    assert metrics["mae"] == 5.0
    assert metrics["rmse"] > 0

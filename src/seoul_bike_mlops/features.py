"""Turn the cleaned data into model features.

Season is one-hot encoded against a fixed category list on purpose: the training
window never sees autumn while the current window is all autumn, so without fixed
categories the two feature tables would end up with different columns.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
# dew point is almost perfectly collinear with temperature, so it's dropped
DROP_AFTER = ["date", "holiday", "functioning_day", "dew_point_temperature"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["functioning_day"] == "Yes"].copy()
    df["date"] = pd.to_datetime(df["date"])

    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_holiday"] = (df["holiday"] == "Holiday").astype(int)
    month = df["date"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    df["season"] = pd.Categorical(df["season"], categories=SEASONS)
    df = pd.get_dummies(df, columns=["season"], prefix="season")
    for col in [f"season_{s}" for s in SEASONS]:
        df[col] = df[col].astype(int)

    return df.drop(columns=DROP_AFTER).reset_index(drop=True)


def main(params_path: str = "params.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = yaml.safe_load(Path(params_path).read_text())["featurize"]

    for src_key, out_key in [("reference_in", "reference_out"), ("current_in", "current_out")]:
        df = pd.read_csv(params[src_key])
        feats = engineer_features(df)
        out = Path(params[out_key])
        out.parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(out, index=False)
        logger.info("%s -> %s (%d rows, %d cols)", params[src_key], out, *feats.shape)


if __name__ == "__main__":
    main()

"""Train the LightGBM demand model and log the run to MLflow.

The holdout is the last slice of the reference window in time order, not a random
split, because this is a time series and a random split would leak future hours
into training.
"""

import logging
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def time_split(df: pd.DataFrame, holdout_fraction: float):
    cut = int(len(df) * (1 - holdout_fraction))
    return df.iloc[:cut], df.iloc[cut:]


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }


def main(params_path: str = "params.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = yaml.safe_load(Path(params_path).read_text())
    cfg, mf = params["train"], params["mlflow"]

    df = pd.read_csv(cfg["features_path"])
    train_df, hold_df = time_split(df, cfg["holdout_fraction"])
    y_train = train_df.pop(cfg["target"])
    y_hold = hold_df.pop(cfg["target"])

    # the target is right-skewed; training on log1p and inverting keeps predictions sane
    y_fit = np.log1p(y_train) if cfg["log_target"] else y_train

    mlflow.set_tracking_uri(mf["tracking_uri"])
    mlflow.set_experiment(mf["experiment"])

    with mlflow.start_run():
        model = lgb.LGBMRegressor(**cfg["lgbm"])
        model.fit(train_df, y_fit)

        pred = model.predict(hold_df)
        if cfg["log_target"]:
            pred = np.expm1(pred)
        pred = np.clip(pred, 0, None)

        metrics = evaluate(y_hold, pred)
        mlflow.log_params(cfg["lgbm"])
        mlflow.log_param("log_target", cfg["log_target"])
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(params_path)
        mlflow.lightgbm.log_model(model, name="model")

        model_out = Path(cfg["model_out"])
        model_out.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(model_out)

    logger.info("holdout MAE=%.1f RMSE=%.1f R2=%.3f", metrics["mae"], metrics["rmse"], metrics["r2"])


if __name__ == "__main__":
    main()

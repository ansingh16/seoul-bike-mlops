"""Score the saved model on the holdout and write metrics.json.

Training already logs metrics to MLflow, but DVC wants a plain file it can track
in git so `dvc metrics diff` works across commits. So this stage reloads the model
booster and re-scores the same holdout slice train.py used.
"""

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from seoul_bike_mlops.train import evaluate, time_split

logger = logging.getLogger(__name__)


def main(params_path: str = "params.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = yaml.safe_load(Path(params_path).read_text())
    cfg = params["train"]
    out_path = Path(params["evaluate"]["metrics_out"])

    df = pd.read_csv(cfg["features_path"])
    _, hold_df = time_split(df, cfg["holdout_fraction"])
    y_hold = hold_df.pop(cfg["target"])

    booster = lgb.Booster(model_file=cfg["model_out"])
    pred = booster.predict(hold_df)
    if cfg["log_target"]:
        pred = np.expm1(pred)
    pred = np.clip(pred, 0, None)

    metrics = evaluate(y_hold, pred)
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    logger.info("holdout MAE=%.1f RMSE=%.1f R2=%.3f", metrics["mae"], metrics["rmse"], metrics["r2"])


if __name__ == "__main__":
    main()

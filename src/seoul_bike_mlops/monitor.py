"""Compare the current window against the reference window and flag drift.

The current window is later in time (and here it's all autumn, which the model never
trained on), so this is where input drift and any performance drop should show up.
Writes an Evidently HTML report and exits non-zero if too large a share of columns
drifted -- something a CI job or a cron can act on.
"""

import json
import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.presets import DataDriftPreset, RegressionPreset

from seoul_bike_mlops.features import engineer_features

logger = logging.getLogger(__name__)

NUMERICAL = ["hour", "temperature", "humidity", "wind_speed", "visibility",
             "solar_radiation", "rainfall", "snowfall"]
CATEGORICAL = ["season", "holiday"]
TARGET = "rented_bike_count"


def load_model(model_dir: str):
    model = mlflow.lightgbm.load_model(model_dir)
    info = json.loads(Path(model_dir, "info.json").read_text())
    booster = model.booster_ if hasattr(model, "booster_") else model
    return model, booster.feature_name(), info["log_target"]


def score(df: pd.DataFrame, model, feature_names, log_target: bool) -> np.ndarray:
    feats = engineer_features(df).reindex(columns=feature_names, fill_value=0)
    pred = model.predict(feats)
    if log_target:
        pred = np.expm1(pred)
    return np.clip(pred, 0, None)


def drifted_share(result) -> tuple[int, float]:
    for m in result.dict()["metrics"]:
        if m["config"]["type"].endswith("DriftedColumnsCount"):
            return int(m["value"]["count"]), float(m["value"]["share"])
    raise RuntimeError("drift summary metric not found in report")


def main(params_path: str = "params.yaml") -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(params_path).read_text())["monitor"]

    model, feature_names, log_target = load_model(cfg["model_dir"])
    frames = {}
    for key in ("reference_in", "current_in"):
        df = pd.read_csv(cfg[key])
        df = df[df["functioning_day"] == "Yes"].reset_index(drop=True)
        keep = df[NUMERICAL + CATEGORICAL + [TARGET]].copy()
        keep["prediction"] = score(df, model, feature_names, log_target)
        frames[key] = keep

    data_def = DataDefinition(
        numerical_columns=NUMERICAL,
        categorical_columns=CATEGORICAL,
        regression=[Regression(target=TARGET, prediction="prediction")],
    )
    ref_ds = Dataset.from_pandas(frames["reference_in"], data_definition=data_def)
    cur_ds = Dataset.from_pandas(frames["current_in"], data_definition=data_def)

    report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
    result = report.run(current_data=cur_ds, reference_data=ref_ds)

    out = Path(cfg["report_out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(out))

    count, share = drifted_share(result)
    logger.info("%d columns drifted (%.0f%% of monitored features) -> %s", count, share * 100, out)

    if share > cfg["drift_threshold"]:
        logger.warning("drift share %.2f over threshold %.2f -- retraining warranted",
                       share, cfg["drift_threshold"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

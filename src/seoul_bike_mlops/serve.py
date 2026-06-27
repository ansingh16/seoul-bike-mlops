"""FastAPI service that serves the current champion model.

Two ways to get the model. If MODEL_DIR points at a self-contained model export
(what the container ships with), load that -- no tracking DB, no absolute paths.
Otherwise fall back to the registry's 'champion' alias, which is handy in local dev
where the MLflow store is right there. Either way, raw weather observations go
through the same feature engineering as training to avoid skew.
"""

import json
import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from seoul_bike_mlops.features import engineer_features

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.environ.get("MODEL_NAME", "seoul-bike-demand")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")

# filled in at startup by load_champion()
STATE: dict = {"model": None, "info": {}, "features": []}


class Observation(BaseModel):
    date: str = Field(examples=["01/12/2018"])
    hour: int = Field(ge=0, le=23)
    temperature: float
    humidity: int = Field(ge=0, le=100)
    wind_speed: float = Field(ge=0)
    visibility: int = Field(ge=0)
    dew_point_temperature: float
    solar_radiation: float = Field(ge=0)
    rainfall: float = Field(ge=0)
    snowfall: float = Field(ge=0)
    season: str
    holiday: str = "No Holiday"
    functioning_day: str = "Yes"


def _load_from_dir(path: str):
    model = mlflow.lightgbm.load_model(path)
    info = json.loads(Path(path, "info.json").read_text())
    logger.info("loaded champion from %s (v%s)", path, info.get("version"))
    return model, info


def _load_from_registry():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    run = client.get_run(mv.run_id)
    info = {
        "name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": mv.version,
        "log_target": run.data.params.get("log_target") == "True",
        "rmse": run.data.metrics.get("rmse"),
    }
    logger.info("loaded %s v%s from registry", MODEL_NAME, mv.version)
    return model, info


def load_champion() -> None:
    if MODEL_DIR and Path(MODEL_DIR).exists():
        model, info = _load_from_dir(MODEL_DIR)
    else:
        model, info = _load_from_registry()

    booster = model.booster_ if hasattr(model, "booster_") else model
    STATE["model"] = model
    STATE["features"] = booster.feature_name()
    STATE["info"] = info


def predict_count(obs: Observation) -> float:
    # a closed system has no rentals; the model never saw those rows, so don't ask it
    if obs.functioning_day != "Yes":
        return 0.0
    row = pd.DataFrame([obs.model_dump()])
    feats = engineer_features(row).reindex(columns=STATE["features"], fill_value=0)
    pred = STATE["model"].predict(feats)
    if STATE["info"]["log_target"]:
        pred = np.expm1(pred)
    return float(np.clip(pred, 0, None)[0])


app = FastAPI(title="Seoul bike demand")


@app.on_event("startup")
def _startup() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_champion()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": STATE["model"] is not None}


@app.get("/model-info")
def model_info() -> dict:
    if STATE["model"] is None:
        raise HTTPException(503, "model not loaded")
    return STATE["info"]


@app.post("/predict")
def predict(obs: Observation) -> dict:
    if STATE["model"] is None:
        raise HTTPException(503, "model not loaded")
    return {"predicted_rented_bike_count": round(predict_count(obs), 1)}

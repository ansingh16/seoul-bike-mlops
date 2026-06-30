# seoul-bike-mlops

![ci](https://github.com/ansingh16/seoul-bike-mlops/actions/workflows/ci.yml/badge.svg)

Predict how many bikes get rented per hour in Seoul from weather and calendar
features. The model itself is deliberately small — the repo is really about the
workflow around it: reproducible data and training stages, experiment tracking, a
gated model registry, a served prediction API, and drift monitoring, all running
locally on a laptop with no cloud dependencies.

Data is the [Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)
set from UCI: 8,760 hourly records spanning Dec 2017 to Nov 2018.

## What's in here

```
                ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌───────┐   ┌──────────┐
   UCI zip ───▶ │  ingest  │─▶ │ validate │─▶ │ featurize │─▶ │ train │─▶ │ evaluate │─▶ metrics.json
                └──────────┘   └──────────┘   └───────────┘   └───┬───┘   └──────────┘
                  (DVC pipeline)                                  │
                                                                  ▼
                                                           ┌────────────┐
                                                           │   MLflow   │  runs + registry
                                                           │  registry  │  (champion alias)
                                                           └─────┬──────┘
                                                                 │ export
                                                                 ▼
                                       serving_model/  ──▶  FastAPI + Docker  ──▶  /predict
                                                                 │
                                                                 ▼
                                                        Evidently drift report
                                                        (exit 1 → retrain signal)
```

| Concern              | Tool                                  |
|----------------------|---------------------------------------|
| Pipeline / repro     | DVC                                   |
| Data validation      | Pandera                               |
| Model                | LightGBM                              |
| Experiment tracking  | MLflow (SQLite backend)               |
| Model registry       | MLflow, gated `champion` alias        |
| Serving              | FastAPI + Docker (multi-stage)        |
| Drift monitoring     | Evidently                             |
| CI                   | GitHub Actions (ruff + pytest)        |

## Quickstart

```
pip install -e ".[dev,serve,monitor]"
dvc repro                              # ingest -> validate -> featurize -> train -> evaluate
python -m seoul_bike_mlops.registry    # register the run, promote/keep champion, export it
uvicorn seoul_bike_mlops.serve:app --port 8000
```

The common commands are also wrapped as `make` targets — `make repro`,
`make serve`, `make monitor`, `make test`. A full step-by-step tour with real
output is in [DEMO.md](DEMO.md).

## The model

LightGBM regressor on cyclical time features (hour and month encoded as sin/cos),
weekend/holiday flags, weather, and one-hot season. The target is right-skewed, so
it's trained on `log1p(count)` and inverted at predict time. The holdout is the
last 20% of the training window in time order — not a random split, since a random
split would leak future hours into training.

Holdout performance:

| metric | value |
|--------|-------|
| MAE    | ~236  |
| RMSE   | ~332  |
| R²     | ~0.69 |

Nothing state-of-the-art, and that's fine — the point is the machinery around it.
Notably, on the later autumn window (a season the model never trained on) RMSE
climbs to ~360, which is exactly what the drift monitor is there to catch.

## Registry and the promotion gate

`registry` registers each training run as a new model version and only moves the
`champion` alias if the new version's holdout RMSE beats the incumbent. A worse or
equal retrain stays registered as a candidate but doesn't take over serving — so a
bad run can't silently replace a model that's already in production. Every run
re-exports the current champion to `serving_model/`, a self-contained model dir
(with `info.json`) that the serving image ships with, so the container needs
neither the tracking DB nor any absolute artifact paths.

## Serving

`seoul_bike_mlops.serve` is a FastAPI app with `/health`, `/model-info`, and
`/predict`. It loads the baked-in `serving_model/` export when `MODEL_DIR` is set
(the container path) and otherwise falls back to the registry's `champion` alias
(handy in local dev). Incoming observations go through the same feature engineering
as training, so there's no train/serve skew. Build and run the container with:

```
docker compose up --build
```

## Monitoring

`seoul_bike_mlops.monitor` scores the reference and current windows with the
champion, builds an Evidently report (`reports/drift.html`) covering both input
drift and regression quality, and exits non-zero when the drifted-feature share
crosses the threshold in `params.yaml`. That non-zero exit is the hook a CI job or
cron would use to kick off a retrain. The notebook `notebooks/02_monitoring.ipynb`
walks through the same analysis with the plots inline.

## Tests and CI

```
make test     # pytest
make lint     # ruff
```

The suite covers feature engineering, the validation schema, the train/eval
helpers, and the serving API (against a tiny in-memory model, so it runs without
the real champion export). GitHub Actions runs ruff and pytest on every push and
pull request.

## Layout

```
src/seoul_bike_mlops/   ingest, validate, features, train, evaluate, registry, serve, monitor
tests/                  unit + api tests
notebooks/              01_eda, 02_monitoring
dvc.yaml, params.yaml   pipeline definition and all config
Dockerfile, docker-compose.yml
```

Config lives entirely in `params.yaml` — paths, the split date, LightGBM
hyperparameters, and the drift threshold — so nothing is hard-coded in the source.

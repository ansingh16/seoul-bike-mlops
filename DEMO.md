# Walkthrough

A guided tour of the pipeline end to end: reproduce the data and model, look at
the tracked runs, promote a champion, serve predictions, and run a drift check.
All commands are from the repo root. Outputs below are copied from a real run so
you know what to expect.

## 0. Setup

```
pip install -e ".[dev,serve,monitor]"
```

Everything after this is also available as `make` targets (`make repro`,
`make serve`, `make monitor`, `make test`, ...).

## 1. Reproduce the pipeline

DVC runs the stages in order and skips any whose inputs haven't changed.

```
dvc repro
```

The stages are `ingest -> validate -> featurize -> train -> evaluate`. `ingest`
downloads the UCI dataset and splits it into a reference window (Dec 2017 – Aug
2018, used for training) and a later current window (Sep – Nov 2018, held back for
drift checks). `evaluate` writes the holdout metrics as a DVC metric to
`metrics.json`:

```
{
  "mae": 236.47,
  "rmse": 331.70,
  "r2": 0.6930
}
```

`dvc metrics show` renders the same file, and `dvc repro` re-runs only the stages
whose inputs changed.

## 2. Experiment tracking

Training logs params, metrics, and the model to MLflow (SQLite-backed, so there's
nothing to stand up). To browse the runs:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://127.0.0.1:5000
```

## 3. Register and promote a champion

`registry` registers the latest run as a new model version, then decides whether
it should become the serving champion. The rule: it only takes the `champion`
alias if its holdout RMSE beats the current champion. A worse (or equal) retrain
gets registered as a candidate but the alias doesn't move, so it can't silently
replace whatever is already serving.

```
python -m seoul_bike_mlops.registry
```

```
INFO registered seoul-bike-demand v4 (rmse=331.7)
INFO kept champion v1 (331.7); v4 did not improve (331.7)
INFO exported champion v1 to serving_model/
```

That's the guard working: v4 didn't beat the incumbent, so v1 stays champion. The
first time you ever run it, there's no incumbent and you'd see instead:

```
INFO registered seoul-bike-demand v1 (rmse=331.7)
INFO no incumbent -- v1 is the first champion
```

and if a genuinely better model were registered:

```
INFO promoted v5: 305.2 beats champion v1 331.7
```

The last line of every run re-exports the current champion to `serving_model/` — a
self-contained model directory with an `info.json`. That's what the serving image
ships with, so the container never needs the MLflow store or any absolute paths.

## 4. Serve predictions

### Locally

```
uvicorn seoul_bike_mlops.serve:app --port 8000
```

With no `MODEL_DIR` set, the API loads the `champion` alias straight from the
registry — handy in dev. In the container it loads the baked-in `serving_model/`
export instead.

```
curl -s localhost:8000/model-info
```

```
{"name":"seoul-bike-demand","alias":"champion","version":1,"log_target":true,"rmse":331.70243319817735}
```

A prediction takes a raw weather + calendar observation (the same shape as the raw
data) and runs it through the identical feature engineering used in training:

```
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "date":"01/12/2018","hour":18,"temperature":-2.0,"humidity":40,
  "wind_speed":1.5,"visibility":2000,"dew_point_temperature":-12.0,
  "solar_radiation":0.0,"rainfall":0.0,"snowfall":0.0,"season":"Winter"}'
```

```
{"predicted_rented_bike_count":567.7}
```

A warm summer morning predicts much higher (`hour=8, season=Summer, temperature=23`
→ ~2105), and a non-functioning day short-circuits to `0.0` without troubling the
model, since it never saw those rows in training.

### In a container

```
docker compose up --build
```

The image is a multi-stage build: dependencies go into a virtualenv in the builder
stage, then the final slim image copies that venv plus the `serving_model/` export
and runs as a non-root user. `MODEL_DIR=/app/serving_model` is baked in, so the
container is self-contained — no MLflow DB, no network. The compose file wires a
health check against `/health`.

## 5. Drift monitoring

The current window is all autumn — a season the model never trained on — so it's a
natural stress test. `monitor` scores both windows with the champion, builds an
Evidently report comparing them, and **exits non-zero** if too large a share of
features have drifted (the threshold lives in `params.yaml`). That exit code is
what a CI job or cron would trigger a retrain on.

```
python -m seoul_bike_mlops.monitor; echo "exit: $?"
```

```
INFO 5 columns drifted (50% of monitored features) -> reports/drift.html
WARNING drift share 0.50 over threshold 0.30 -- retraining warranted
exit: 1
```

Open `reports/drift.html` for the full breakdown: season shifts hardest
(everything is autumn now), temperature and wind speed follow, and the regression
quality section shows RMSE degrading badly on this unseen window — exactly the
signal you'd want a retrain to fire on.

The exploratory version of this, with the plots inline, is in
`notebooks/02_monitoring.ipynb`.

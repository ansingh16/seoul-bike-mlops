# seoul-bike-mlops

Predict how many bikes get rented per hour in Seoul from weather and calendar
features. The model is deliberately small — the repo is really about the workflow
around it: reproducible data and training stages, experiment tracking, a model
registry, a served prediction API, and drift monitoring, all running locally on a
laptop.

Data is the [Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)
set from UCI: 8,760 hourly records spanning Dec 2017 to Nov 2018.

## Status

Work in progress. Done so far:

- DVC `ingest` stage — downloads the data, cleans the column names, and splits it
  into a reference window (Dec 2017 – Aug 2018) for training and a later current
  window (Sep – Nov 2018) used for drift checks.
- An EDA notebook with a first look at the data and a quick baseline.

## Setup

```
pip install -e ".[dev]"
dvc repro          # downloads the data and builds the reference/current split
```

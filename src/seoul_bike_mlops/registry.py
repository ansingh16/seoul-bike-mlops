"""Register the most recent training run's model in the MLflow registry and point
the 'champion' alias at it."""

import logging
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def latest_run(client: MlflowClient, experiment: str):
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise RuntimeError(f"experiment {experiment!r} does not exist yet -- run training first")
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise RuntimeError(f"no runs found in {experiment!r}")
    return runs[0]


def main(params_path: str = "params.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    mf = yaml.safe_load(Path(params_path).read_text())["mlflow"]
    mlflow.set_tracking_uri(mf["tracking_uri"])
    client = MlflowClient()
    name = mf["registered_model"]

    run = latest_run(client, mf["experiment"])
    version = mlflow.register_model(f"runs:/{run.info.run_id}/model", name)
    logger.info("registered %s v%s from run %s", name, version.version, run.info.run_id[:8])

    client.set_registered_model_alias(name, "champion", version.version)
    logger.info("champion -> v%s", version.version)


if __name__ == "__main__":
    main()

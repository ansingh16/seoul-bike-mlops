"""Register the latest training run and decide whether it becomes champion.

A new model only takes over the 'champion' alias if it beats the incumbent on the
holdout RMSE that was logged at training time. Otherwise it stays registered as a
candidate version but the alias doesn't move -- so a worse retrain can't silently
replace a model that's already serving.
"""

import logging
from pathlib import Path

import mlflow
import yaml
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

METRIC = "rmse"  # lower is better


def latest_run(client: MlflowClient, experiment: str):
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise RuntimeError(f"experiment {experiment!r} does not exist yet -- run training first")
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise RuntimeError(f"no runs found in {experiment!r}")
    return runs[0]


def champion_version(client: MlflowClient, name: str):
    try:
        return client.get_model_version_by_alias(name, "champion")
    except MlflowException:
        return None


def main(params_path: str = "params.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    mf = yaml.safe_load(Path(params_path).read_text())["mlflow"]
    mlflow.set_tracking_uri(mf["tracking_uri"])
    client = MlflowClient()
    name = mf["registered_model"]

    run = latest_run(client, mf["experiment"])
    new_score = run.data.metrics[METRIC]
    version = mlflow.register_model(f"runs:/{run.info.run_id}/model", name)
    logger.info("registered %s v%s (%s=%.1f)", name, version.version, METRIC, new_score)

    champ = champion_version(client, name)
    if champ is None:
        client.set_registered_model_alias(name, "champion", version.version)
        logger.info("no incumbent -- v%s is the first champion", version.version)
        return

    champ_score = client.get_run(champ.run_id).data.metrics[METRIC]
    if new_score < champ_score:
        client.set_registered_model_alias(name, "champion", version.version)
        logger.info("promoted v%s: %.1f beats champion v%s %.1f", version.version,
                    new_score, champ.version, champ_score)
    else:
        logger.info("kept champion v%s (%.1f); v%s did not improve (%.1f)", champ.version,
                    champ_score, version.version, new_score)


if __name__ == "__main__":
    main()

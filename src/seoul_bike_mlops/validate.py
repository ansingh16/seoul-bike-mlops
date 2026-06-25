"""Schema and range checks on the processed data, run as a pipeline stage so a
bad ingest fails loudly before anything downstream trains on it."""

import logging
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import yaml

logger = logging.getLogger(__name__)

SEASONS = ["Spring", "Summer", "Autumn", "Winter"]

SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column("datetime64[ns]", coerce=True),
        "rented_bike_count": pa.Column(int, pa.Check.ge(0)),
        "hour": pa.Column(int, pa.Check.in_range(0, 23)),
        "temperature": pa.Column(float, pa.Check.in_range(-30, 45)),
        "humidity": pa.Column(int, pa.Check.in_range(0, 100)),
        "wind_speed": pa.Column(float, pa.Check.ge(0)),
        "visibility": pa.Column(int, pa.Check.ge(0)),
        "dew_point_temperature": pa.Column(float),
        "solar_radiation": pa.Column(float, pa.Check.ge(0)),
        "rainfall": pa.Column(float, pa.Check.ge(0)),
        "snowfall": pa.Column(float, pa.Check.ge(0)),
        "season": pa.Column(str, pa.Check.isin(SEASONS)),
        "holiday": pa.Column(str, pa.Check.isin(["Holiday", "No Holiday"])),
        "functioning_day": pa.Column(str, pa.Check.isin(["Yes", "No"])),
    }
)


def validate_file(path: str | Path) -> pd.DataFrame:
    """Load a processed CSV and validate it against the schema."""
    df = pd.read_csv(path, parse_dates=["date"])
    return SCHEMA.validate(df, lazy=True)


def main(params_path: str = "params.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = yaml.safe_load(Path(params_path).read_text())["validate"]

    lines = []
    for path in params["inputs"]:
        df = validate_file(path)
        msg = f"{path}: {len(df)} rows passed validation"
        logger.info(msg)
        lines.append(msg)

    report = Path(params["report_out"])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

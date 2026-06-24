"""Download the Seoul bike-sharing data and split it into a reference window
(for training) and a later current window (for drift monitoring)."""

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)

COLUMN_RENAME = {
    "Date": "date",
    "Rented Bike Count": "rented_bike_count",
    "Hour": "hour",
    "Temperature(°C)": "temperature",
    "Humidity(%)": "humidity",
    "Wind speed (m/s)": "wind_speed",
    "Visibility (10m)": "visibility",
    "Dew point temperature(°C)": "dew_point_temperature",
    "Solar Radiation (MJ/m2)": "solar_radiation",
    "Rainfall(mm)": "rainfall",
    "Snowfall (cm)": "snowfall",
    "Seasons": "season",
    "Holiday": "holiday",
    "Functioning Day": "functioning_day",
}


def download_raw(url: str, dest: Path) -> Path:
    """Fetch the dataset zip from UCI and extract the CSV to dest."""
    if dest.exists():
        logger.info("raw file already present, skipping download")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        dest.write_bytes(zf.read(csv_name))
    return dest


def load_raw(path: Path) -> pd.DataFrame:
    # the file ships with a degree symbol in the headers, so it isn't utf-8
    df = pd.read_csv(path, encoding="latin-1").rename(columns=COLUMN_RENAME)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    return df.sort_values(["date", "hour"]).reset_index(drop=True)


def split_windows(df: pd.DataFrame, split_date: str):
    cut = pd.to_datetime(split_date)
    reference = df[df["date"] < cut].reset_index(drop=True)
    current = df[df["date"] >= cut].reset_index(drop=True)
    return reference, current


def main(params_path: str = "params.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = yaml.safe_load(Path(params_path).read_text())["ingest"]

    raw_csv = Path(params["raw_csv"])
    download_raw(params["raw_url"], raw_csv)
    df = load_raw(raw_csv)

    reference, current = split_windows(df, params["split_date"])
    for out, frame in [(params["reference_out"], reference), (params["current_out"], current)]:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
    logger.info("reference=%d rows, current=%d rows", len(reference), len(current))


if __name__ == "__main__":
    main()

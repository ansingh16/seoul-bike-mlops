.PHONY: install repro register serve monitor test lint docker docker-up clean

install:  ## install the package with dev, serve and monitor extras
	pip install -e ".[dev,serve,monitor]"

repro:  ## run the full dvc pipeline (ingest -> validate -> featurize -> train -> evaluate)
	dvc repro

register:  ## register the latest run and (re)export the champion for serving
	python -m seoul_bike_mlops.registry

serve:  ## run the api locally against the registry champion
	uvicorn seoul_bike_mlops.serve:app --reload --port 8000

monitor:  ## score the current window and write the drift report (exit 1 if drifted)
	python -m seoul_bike_mlops.monitor

test:  ## run the test suite
	pytest -q

lint:  ## lint the source and tests
	ruff check src tests

docker:  ## build the serving image (needs a champion export in serving_model/)
	docker build -t seoul-bike-demand .

docker-up:  ## build and run the api in a container on :8000
	docker compose up --build

clean:  ## remove generated reports and caches
	rm -rf reports/*.html reports/*.txt .ruff_cache .pytest_cache

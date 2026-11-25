.PHONY: install train test lint serve load-test docker-build docker-run

install:
	pip install -r requirements-dev.txt

train:
	PYTHONPATH=src python -m mlpipe.train

test:
	pytest -q

lint:
	ruff check src tests scripts

serve: train
	uvicorn mlpipe.app:app --app-dir src --host 0.0.0.0 --port 8000

load-test:
	python scripts/load_test.py --n 1000

docker-build:
	docker build -t mlops-mini-pipeline:latest .

docker-run:
	docker run --rm -p 8000:8000 mlops-mini-pipeline:latest

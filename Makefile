.PHONY: setup data eda train test lint all serve docker-build

setup:
	python -m pip install -r requirements.txt
	python -m pip install -e .

data:
	python scripts/download_data.py

eda: data
	python scripts/run_eda.py

train: data
	python scripts/train_model.py

test:
	pytest

lint:
	ruff check src tests scripts

all: data
	python scripts/run_eda.py
	python scripts/train_model.py
	pytest

serve:
	uvicorn heart_disease_mlops.api:app --host 0.0.0.0 --port 8000

docker-build: train
	docker build -t heart-disease-mlops:local .
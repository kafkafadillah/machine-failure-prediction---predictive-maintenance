PY ?= python

.PHONY: all eda prepare train evaluate deploy test serve predict clean fast

all: eda prepare train evaluate deploy   ## seluruh siklus CRISP-DM

eda:        ; $(PY) pipelines/02_data_understanding.py
prepare:    ; $(PY) pipelines/03_data_preparation.py
train:      ; $(PY) pipelines/04_modeling.py
fast:       ; $(PY) pipelines/04_modeling.py --fast
evaluate:   ; $(PY) pipelines/05_evaluation.py
deploy:     ; $(PY) pipelines/06_deployment.py
test:       ; $(PY) -m pytest -q
serve:      ; uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
predict:    ; $(PY) app/predict_cli.py --input data/raw/machine_failure.csv --output reports/tables/predictions_all.csv --drift-report reports/tables/drift_all.csv

clean:
	rm -rf data/processed/*.csv data/processed/*.json models/*.joblib models/*.json models/*.csv reports/figures/* reports/tables/*

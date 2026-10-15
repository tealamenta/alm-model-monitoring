.PHONY: install lint types security test check audit demo pipeline monitor notebooks clean

PY ?= python

install:          ## Environnement de développement (uv)
	uv sync --all-extras --group dev

lint:             ## ruff (style + format)
	ruff check .
	ruff format --check .

types:            ## mypy strict
	mypy

security:         ## bandit + pip-audit
	bandit -q -c pyproject.toml -r src scripts dataiku
	pip-audit

test:             ## pytest avec couverture (sans le test PySpark)
	pytest -m "not slow"

check: lint types security test   ## toutes les portes qualité

demo:             ## pipeline complet sur l'échantillon livré (≈ 1 min, sans Kaggle ; fenêtres trimestrielles)
	-$(PY) scripts/audit_data.py --raw data/sample/lending_club_sample.csv.gz \
	    --reports reports/lending_club_demo --name lending_club_sample
	$(PY) scripts/train_lending_club.py --raw data/sample/lending_club_sample.csv.gz \
	    --out data/processed/lending_club_demo --reports reports/lending_club_demo \
	    --models models/demo --quick
	# le code de sortie reflète la gravité (0 ok, 1 warning, 2 critical) : on ne l'interprète pas ici
	-$(PY) scripts/run_monitoring.py --data data/processed/lending_club_demo \
	    --out reports/lending_club_demo --freq QS --no-alerts

pipeline:         ## pipeline complet sur les données réelles (data/raw/accepted_2007_to_2018Q4.csv.gz)
	$(PY) scripts/audit_data.py
	$(PY) scripts/train_lending_club.py
	-$(PY) scripts/run_monitoring.py --no-alerts
	$(PY) scripts/make_readme_figure.py
	$(PY) scripts/export_mlflow_model.py --track

audit:            ## audit de qualité des données seul (verdict d'exploitabilité)
	$(PY) scripts/audit_data.py

monitor:          ## monitoring seul (données déjà préparées) ; code de sortie = gravité
	$(PY) scripts/run_monitoring.py --no-alerts

notebooks:        ## ré-exécute les notebooks
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

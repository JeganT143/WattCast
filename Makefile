PYTHON ?= python

.PHONY: install install-dev app api data test lint

install:        ## runtime dependencies (Streamlit app)
	$(PYTHON) -m pip install -r requirements.txt

install-dev:    ## runtime + API, training, notebooks, tests
	$(PYTHON) -m pip install -r requirements-dev.txt

app:            ## run the Streamlit app on http://localhost:8501
	$(PYTHON) -m streamlit run streamlit_app.py

api:            ## run the REST API on http://127.0.0.1:8000 (docs at /docs)
	$(PYTHON) -m uvicorn src.serving.app:app

data:           ## download the raw dataset (checksum-verified) and rebuild data/processed
	$(PYTHON) -m scripts.download_data
	$(PYTHON) -m src.build_processed --out data/processed

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

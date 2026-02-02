---
title: Automated ML Pipeline with CI/CD
emoji: 🤖
colorFrom: gray
colorTo: red
sdk: docker
app_file: Dockerfile
pinned: false
license: mit
---

# Under Construction

# Notes

Raw dataset: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

py -3.10 -m venv .venv

.\\.venv\Scripts\activate

python -m pip install --upgrade pip

pip install -r requirements-dev.txt

python scripts/bootstrap.py

uvicorn app.main:app --reload

# Automated ML Pipeline with CI/CD

This repository contains a fully automated **Machine Learning pipeline** with **CI/CD capabilities**, designed for **house price prediction in King County, USA, 2015**. The project features reproducible model training, metric-based quality gates, versioned model packaging, and deployment-ready artifacts. It leverages **Python 3.10**, **scikit-learn** and **GitHub Actions** for automation, ensuring enterprise-grade reproducibility and governance.

The system is designed to run locally on venv python 3.10 or on Hugging Face Spaces, with minimal dependencies and a purposely simple front-end.

Hugging Face Space: [LeonardoMdSA / Automated ML Pipeline with CI/CD](https://huggingface.co/spaces/LeonardoMdSA/Automated-ML-Pipeline-with-CI-CD)

---

## Features

* **Automated Training & Evaluation:** Run training and evaluation pipelines with a single command.
* **Model Versioning:** Versioned models stored in a local registry (`models/registry`) with metadata.
* **Quality Gates:** Metric-based evaluation ensures only high-quality models are promoted.
* **Artifact Packaging:** Models packaged with metrics and metadata for reproducibility.
* **CI/CD Pipelines:** Fully automated using GitHub Actions for testing, evaluation, and deployment.
* **Web Interface:** Minimal FastAPI dashboard for predictions.

---

## Repository Structure (After running bootstrap.py)

```
Automated ML Pipeline with CI-CD/
├── .github/
│ └── workflows/
│ ├── deploy_hf.yml # Deploy inference app to Hugging Face Spaces
│ └── ml_pipeline.yml # CI pipeline: train, evaluate, gate, package
├── .vscode/
│ └── settings.json # VS Code workspace settings
├── app/ # Inference service (FastAPI)
│ ├── api/
│ │ └── routes.py # Prediction API routes
│ ├── core/
│ │ ├── config.py # App configuration
│ │ └── logging.py # Structured logging setup
│ ├── inference/
│ │ └── predictor.py # Loads *packaged* model and runs inference
│ ├── schemas/
│ │ └── request_response.py # Pydantic request/response schemas
│ ├── static/
│ │ └── styles.css # Frontend styles
│ ├── templates/
│ │ └── index.html # Minimal HTML frontend
│ └── main.py # FastAPI application entrypoint
├── data/
│ ├── raw/
│ │ └── kc_house_data.csv # Raw dataset
│ └── processed/
│ └── train_test.npz # Train/test split artifacts
├── models/
│ ├── baseline/
│ │ └── metrics.json # Baseline model metrics
│ ├── packaged/ # Production-ready model artifact
│ │ ├── model.pkl # Serialized best model
│ │ ├── metrics.json # Metrics of packaged model
│ │ └── packaged.json # Packaging metadata
│ └── registry/ # Filesystem-based model registry
│ ├── model_v001/
│ │ ├── model.pkl
│ │ └── metadata.json
│ ├── model_v002/
│ │ ├── model.pkl
│ │ └── metadata.json
│ ├── model_v003/
│ │ ├── model.pkl
│ │ └── metadata.json
│ └── latest.json # Pointer to most recent trained model
├── reports/
│ ├── evaluations/ # Per-run evaluation reports
│ │ ├── model_v001_run*.json
│ │ ├── model_v002_run*.json
│ │ └── model_v003_run*.json
│ └── comparison.json # Model comparison results
├── scripts/ # Pipeline execution scripts
│ ├── bootstrap.py # End-to-end local bootstrap
│ ├── train.py # Deterministic model training
│ ├── evaluate.py # Model evaluation
│ ├── compare.py # Compare candidate vs baseline
│ ├── metric_gate.py # Quality gate enforcement
│ ├── package_model.py # Package best model for inference
│ ├── versioning.py # Model version increment logic
│ ├── config.py # Pipeline configuration
│ └── __init__.py
├── tests/
│ ├── integration/ # End-to-end and CI-like tests
│ │ ├── test_api_predict.py
│ │ ├── test_ci_like_flow.py
│ │ ├── test_gate_blocks_regression.py
│ │ ├── test_model_promotion.py
│ │ └── test_train_evaluate_pipeline.py
│ ├── unit/ # Deterministic unit tests
│ │ ├── test_compare_gate_logic.py
│ │ ├── test_compare_self_comparison_guard.py
│ │ ├── test_data_schema.py
│ │ ├── test_evaluate_deterministic.py
│ │ ├── test_metrics_computation.py
│ │ ├── test_metric_gate.py
│ │ ├── test_registry_metadata.py
│ │ ├── test_train_deterministic.py
│ │ ├── test_train_outputs.py
│ │ └── test_version_increment.py
│ ├── conftest.py
│ └── __init__.py
├── Dockerfile # Container ready for Hugging Face Spaces
├── pytest.ini # Pytest configuration
├── requirements.txt # Runtime dependencies
└── repo_structure.py # Utility to print repo tree
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/LeonardoMdSACode/Automated-ML-Pipeline-CI-CD-Clean.git
cd Automated-ML-Pipeline-CI-CD-Clean
```

2. Create a virtual environment and install dependencies:

```bash
py -3.10 -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## Usage

* **Train a model:** `python scripts/train.py`
* **Evaluate a model:** `python scripts/evaluate.py --model models/packaged/model.pkl`
* **Compare models and apply gates:** `python scripts/compare.py`
* **Package model for deployment:** `python scripts/package_model.py`

#### Or just run: `python scripts/bootstrap.py` instead.

1. Run the FastAPI app locally:

```bash
uvicorn app.main:app --reload
```

* **Check API predictions:** Open `http://127.0.0.1:8000` in your browser

---

## Testing

1. Run all tests with pytest:

   ```bash
   pytest -v
   ```

2. Tests will run regardless during CI at github actions.

---

## Technology Stack

* Python 3.10
* FastAPI
* Uvicorn
* scikit-learn
* GitHub Actions (CI/CD)
* Pydantic 2.12.5
* Jinja2
* Pandas / NumPy
* Joblib (for model serialization)
* Docker (for containerized deployment)
* Hugging Face Spaces (deployment)

This project demonstrates a **reproducible, fully automated ML pipeline** with **enterprise-grade CI/CD practices**, suitable for real-world deployment and model governance.


## MIT License

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

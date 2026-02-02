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

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

This repository contains a fully automated **Machine Learning pipeline** with **CI/CD capabilities**, designed for **house price prediction in King County, USA, 2015**. The project features reproducible model training, metric-based quality gates, versioned model packaging, and deployment-ready artifacts. It leverages **Python 3.10**, **scikit-learn**, **DVC**, and **GitHub Actions** for automation, ensuring enterprise-grade reproducibility and governance.

---

## Features

* **Automated Training & Evaluation:** Run training and evaluation pipelines with a single command.
* **Model Versioning:** Versioned models stored in a local registry (`models/registry`) with metadata.
* **Quality Gates:** Metric-based evaluation ensures only high-quality models are promoted.
* **Artifact Packaging:** Models packaged with metrics and metadata for reproducibility.
* **DVC Integration:** Track datasets, preprocessed data, and model artifacts.
* **CI/CD Pipelines:** Fully automated using GitHub Actions for testing, evaluation, and deployment.
* **Web Interface:** Minimal FastAPI dashboard for predictions.

---

## Repository Structure

```
Automated-ML-Pipeline-CI-CD-Clean/
├─ .dvc/                   # DVC configuration and cache
│  ├─ cache/               # DVC cache files
│  └─ tmp/                 # Temporary DVC files
├─ .github/workflows/      # GitHub Actions workflows
│  ├─ ml_pipeline.yml      # CI/CD pipeline workflow
│  └─ deploy_hf.yml        # Deployment to Hugging Face workflow
├─ .vscode/                # VSCode workspace settings
├─ app/                    # FastAPI application
│  ├─ api/                 # API routes
│  │  └─ routes.py
│  ├─ core/                # Configs and logging
│  │  ├─ config.py
│  │  └─ logging.py
│  ├─ inference/           # Model inference logic
│  │  └─ predictor.py
│  ├─ schemas/             # Request/response schemas
│  │  └─ request_response.py
│  ├─ static/              # CSS and static assets
│  │  └─ styles.css
│  ├─ templates/           # HTML templates
│  │  └─ index.html
│  └─ main.py              # FastAPI app entrypoint
├─ data/                   # Data folder
│  ├─ processed/           # Processed datasets (train/test split)
│  │  └─ train_test.npz
│  ├─ raw/                 # Raw datasets
│  │  └─ kc_house_data.csv
│  └─ reference/           # Reference or lookup data
├─ models/                 # Models and registry
│  ├─ baseline/            # Baseline model and metrics
│  ├─ packaged/            # Packaged model artifacts
│  └─ registry/            # Versioned model registry
│     ├─ model_v001/
│     ├─ model_v002/
│     ├─ model_v003/
│     └─ latest.json       # Points to latest version
├─ reports/                # Evaluation and comparison reports
│  ├─ evaluations/         # Individual model evaluation JSON
│  └─ comparison.json      # Overall model comparison
├─ scripts/                # Pipeline scripts
│  ├─ bootstrap.py         # Initialize environment and dataset
│  ├─ train.py             # Train a new model
│  ├─ evaluate.py          # Evaluate a trained model
│  ├─ compare.py           # Compare models and check quality gates
│  ├─ metric_gate.py       # Metric gate logic
│  ├─ package_model.py     # Package model artifacts
│  ├─ versioning.py        # Manage model versioning
│  └─ config.py            # Pipeline configuration
├─ tests/                  # Tests
│  ├─ unit/                # Unit tests for functions and scripts
│  └─ integration/         # Integration tests for pipeline and API
├─ Dockerfile              # Docker image definition
├─ dvc.yaml                # DVC pipeline definition
├─ requirements.txt        # Production dependencies
├─ requirements-dev.txt    # Development dependencies
├─ pytest.ini              # Pytest configuration
├─ repo_structure.py       # Script to visualize repo structure
└─ README.md               # Project documentation
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
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

3. Pull DVC data:

```bash
dvc pull
```

4. Run the FastAPI app locally:

```bash
uvicorn app.main:app --reload
```

---

## Usage

* **Train a model:** `python scripts/train.py`
* **Evaluate a model:** `python scripts/evaluate.py --model models/packaged/model.pkl`
* **Compare models and apply gates:** `python scripts/compare.py`
* **Package model for deployment:** `python scripts/package_model.py`

### Or run: `python scripts/bootstrap.py` instead.

* **Check API predictions:** Open `http://127.0.0.1:8000` in your browser

---

## Built With

* Python 3.10
* FastAPI
* scikit-learn
* GitHub Actions
* Pydantic

This project demonstrates a **reproducible, fully automated ML pipeline** with **enterprise-grade CI/CD practices**, suitable for real-world deployment and model governance.

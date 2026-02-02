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

## Logical Architecture Overview

This project is organized as a **strictly layered, contract-driven ML system**. Each layer has a single responsibility and clear boundaries. Only gated, packaged artifacts are allowed to reach production.

---

### 1. Data Layer
**Responsibility:** Provide deterministic, version-stable inputs.

- `data/raw/` – immutable source dataset  
- `data/processed/` – canonical train/test split (`train_test.npz`)
- All training and evaluation consume processed data only

---

### 2. Training Layer
**Responsibility:** Produce reproducible candidate models.

- `scripts/train.py` – deterministic model training
- `scripts/versioning.py` – semantic model versioning (`model_vXXX`)
- Outputs candidate models to `models/registry/`

---

### 3. Evaluation Layer
**Responsibility:** Measure model performance in isolation.

- `scripts/evaluate.py` – computes metrics on test data
- `reports/evaluations/` – immutable evaluation artifacts per run
- No comparison or promotion logic at this stage

---

### 4. Comparison & Quality Gate Layer
**Responsibility:** Enforce model quality and prevent regressions.

- `scripts/compare.py` – candidate vs baseline / previous best
- `scripts/metric_gate.py` – hard metric thresholds
- `models/baseline/` – reference metrics
- Any regression blocks promotion

---

### 5. Model Registry Layer
**Responsibility:** Preserve all approved historical models.

- `models/registry/model_vXXX/`
  - `model.pkl`
  - `metadata.json`
- `models/registry/latest.json` – pointer to latest approved model
- Append-only, never used directly by inference

---

### 6. Packaging Layer (Production Boundary)
**Responsibility:** Create the only deployable artifact.

- `scripts/package_model.py` – freezes the approved model
- `models/packaged/`
  - `model.pkl`
  - `metrics.json`
  - `packaged.json`
- Model used in production

---

### 7. Inference Layer
**Responsibility:** Serve predictions from a frozen artifact.

- `app/inference/predictor.py`
- Loads `models/packaged/model.pkl`
- Stateless and read-only

---

### 8. API Layer
**Responsibility:** Expose inference via a stable contract.

- `app/api/routes.py` – `/api/predict`
- `app/schemas/request_response.py` – strict validation
- No ML logic in routes

---

### 9. Application Core Layer
**Responsibility:** Cross-cutting concerns.

- `app/core/config.py` – configuration
- `app/core/logging.py` – structured logging
- No business or ML logic

---

### 10. Presentation Layer (UI)
**Responsibility:** Human interaction only.

- `app/templates/index.html`
- `app/static/styles.css`
- Communicates exclusively via the API

---

### 11. Testing Layer
**Responsibility:** Enforce contracts and prevent regressions.

- `tests/unit/` – deterministic behavior, metrics, versioning
- `tests/integration/` – end-to-end pipeline and CI-like flows
- Every layer boundary is tested

---

### 12. CI/CD Orchestration Layer
**Responsibility:** Automation and governance.

- `.github/workflows/ml_pipeline.yml` – train → evaluate → gate → package
- `.github/workflows/deploy_hf.yml` – deploy packaged model
- No manual promotion or deployment

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

---

## Recommendations & Important Notes

### Model Usage (Critical)
- **Inference always uses the packaged model** located in `models/packaged/model.pkl`.
- The **model registry (`models/registry/`) is append-only and NOT used by the API**.
- Never point inference to `latest.json` or directly to registry versions.
- The packaged model represents the **best model after metric gating**.

---

### CI/CD Guarantees
- Model promotion is **fully automated** and governed by metric gates.
- Any performance regression (e.g., RMSE increase) **blocks packaging and deployment**.
- Manual model promotion is intentionally unsupported.

---

### Determinism & Reproducibility
- Training, evaluation, and comparison are **deterministic by design**.
- Fixed random seeds and immutable processed datasets are enforced.
- Re-running the pipeline with identical inputs produces identical artifacts.

---

### Tests Are First-Class Citizens
- Unit tests validate:
  - Metric correctness
  - Version increments
  - Registry metadata integrity
  - Gate logic
- Integration tests simulate:
  - Full CI-like flows
  - Model regression blocking
  - API inference against packaged models
- Removing or weakening tests undermines pipeline guarantees.

---

### Registry vs Packaged Model (Design Intent)
- `models/registry/` is for **traceability and auditability**.
- `models/packaged/` is the **single production boundary**.
- This separation prevents accidental deployment of unvalidated models.

---

### DVC Removal (Intentional)
- DVC was removed to keep the pipeline:
  - Free-tier compatible
  - Self-contained
  - Fully reproducible via filesystem artifacts
- Data versioning is handled via **immutable processed datasets** and CI enforcement.

---

### Frontend Scope
- The UI is intentionally minimal and **not a data science tool**.
- Its purpose is:
  - Demonstrate inference
  - Display prediction + model version
- All business logic lives in the backend.

---

### Deployment Expectations
- The service is designed to run:
  - Locally via Docker
  - In CI
  - On Hugging Face Spaces (CPU, free tier)
- No cloud credentials or paid services are required.

---

### Extensibility Notes
- Drift detection, monitoring, and alerting can be added **without modifying**:
  - Training
  - Evaluation
  - Packaging logic
- The architecture supports post-deployment observability as a separate layer.

---

### Anti-Patterns (Do Not Do This)
- ❌ Loading models directly from `models/registry/`
- ❌ Skipping metric gates
- ❌ Manually copying models into `packaged/`
- ❌ Training inside the API service
- ❌ Treating `latest.json` as production truth

---

### Intended Audience
- This project is designed to demonstrate:
  - Production-grade ML engineering
  - CI-governed model promotion
  - Deterministic ML pipelines
- It is **not** a notebook-centric or experimentation-only project.

---

## References / Documentation

### Core Technologies
- **FastAPI** – High-performance Python web framework for ML inference APIs  
  https://fastapi.tiangolo.com/

- **Uvicorn** – ASGI server used for running FastAPI services  
  https://www.uvicorn.org/

- **scikit-learn** – Model training, evaluation, and serialization  
  https://scikit-learn.org/stable/

- **NumPy** – Numerical computing and deterministic data handling  
  https://numpy.org/doc/

---

### Testing & Quality
- **pytest** – Unit and integration testing framework  
  https://docs.pytest.org/

- **pytest-cov** – Test coverage reporting  
  https://pytest-cov.readthedocs.io/

---

### MLOps & Engineering Practices
- **ML Test Score (Google)** – Testing levels for ML systems  
  https://research.google/pubs/pub46555/

- **Hidden Technical Debt in ML Systems** – Foundational ML engineering paper  
  https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf

- **Effective Model Versioning** – Practical guidance on model registries and promotion  
  https://martinfowler.com/articles/machine-learning-model-versioning.html

---

### CI/CD & Automation
- **GitHub Actions** – CI/CD workflows for training, testing, and deployment  
  https://docs.github.com/en/actions

- **Semantic Versioning** – Versioning principles applied to model artifacts  
  https://semver.org/

---

### Deployment
- **Docker** – Containerized deployment for reproducible inference services  
  https://docs.docker.com/

- **Hugging Face Spaces** – Free-tier deployment target for ML applications  
  https://huggingface.co/docs/hub/spaces

---

### Dataset
- **King County House Sales Dataset (2014–2015)**  
  https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

---

### Design Philosophy
- **Twelve-Factor App** – Principles for production-ready services  
  https://12factor.net/

- **Separation of Concerns** – Architectural principle applied to training, evaluation, and inference  
  https://en.wikipedia.org/wiki/Separation_of_concerns

---
## Contact / Author

* Hugging Face: [https://huggingface.co/LeonardoMdSA](https://huggingface.co/LeonardoMdSA)
* GitHub: [https://github.com/LeonardoMdSACode](https://github.com/LeonardoMdSACode)

---

## MIT License

This project is licensed under the MIT License. See the LICENSE file for details.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

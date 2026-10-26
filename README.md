# 🦥 SlothML

SlothML is a zero-training tabular machine learning web application built with FastAPI and TabPFN (Tabular Prior-data Fitted Networks). It automates data ingestion, preprocessing, target detection, benchmarking, and real-time inference without requiring traditional model training loops or hyperparameter tuning.

[![Live Demo](https://img.shields.io/badge/🚀_Live_App-SlothML_on_Render-success?style=flat-square&link=https://slothml.onrender.com)](https://slothml.onrender.com)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-005571?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)

---

## 🚀 Live Application
Access the production deployment here: **[https://slothml.onrender.com](https://slothml.onrender.com)**

---

## 📋 Core Features

* **In-Context Tabular Foundation AI (`tabpfn`):** Leverages pre-trained transformer models for fast tabular classification and regression without iterative gradient descent training.
* **Automated Data Pipeline & Sanitization:** Robust ingestion engine that handles missing values, strip whitespace, cleans column headers containing special characters/spaces, and auto-detects data types.
* **Auto-Target & Task Detection:** Automatically infers target/label columns and classifies problem types (binary/multiclass classification vs. regression).
* **Head-to-Head Benchmarking:** Compares TabPFN inference speed and accuracy directly against traditional Random Forest baselines.
* **Interactive "What-If" Simulator:** Real-time inference endpoint allowing users to modify feature inputs and calculate dynamic probability updates.
* **Pre-Loaded Demo Datasets:** Instant evaluation pipelines for SaaS Churn, Real Estate Pricing, and Healthcare Risk assessment.

---

## 🛠️ Tech Stack

* **Backend & API:** Python 3.13, FastAPI, Uvicorn, Pandas, Scikit-Learn
* **Machine Learning Engine:** `tabpfn`
* **Frontend:** Single-Page Application (HTML5, Tailwind CSS, JavaScript, Chart.js, Lucide Icons)
* **Deployment & Testing:** Render (Cloud Platform), Pytest / FastAPI TestClient

---

## 📂 Project Structure


SlothML/
├── app.py              # FastAPI application server & API routes
├── ml_engine.py        # Core machine learning & TabPFN inference pipeline
├── requirements.txt    # Python dependencies
├── Procfile            # Render deployment process configuration
├── run.bat             # Windows local startup script
├── static/             # Frontend assets (HTML, CSS, JS)
└── datasets/           # Pre-loaded demo CSV datasets
⚙️ Getting Started & Installation
Prerequisites
Python 3.13+ installed on your system.

Git for version control.

Local Setup
Clone the repository:

Bash
git clone [https://github.com/joey441/SlothML.git]
cd SlothML
Create and activate a virtual environment (recommended):

Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Install dependencies:

Bash
pip install -r requirements.txt
Run the application:

Bash
python app.py
(Alternatively, execute run.bat on Windows).

Access the application locally:

Open your browser and navigate to http://localhost:8000.

🔌 API Endpoints
POST /api/upload — Accepts multipart CSV file uploads, parses dataframe, and executes schema sanitization.

POST /api/benchmark — Triggers comparative model benchmarking between TabPFN and baseline estimators.

POST /api/simulate — Evaluates real-time "What-If" feature perturbations and returns localized prediction probabilities.

👤 Author
Developed by Joey (joey441)

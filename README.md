# Fake News Classifier

> **Status:** Completed | **Accuracy:** 97% | **Live Demo:** [Available Locally]

A full-stack machine learning application that detects whether a news article is **REAL** or **FAKE** using Natural Language Processing (NLP).

![demo1](https://github.com/user-attachments/assets/f8df011c-5818-4bf6-b5b3-769b19b2e0f4)

![demo2](https://github.com/user-attachments/assets/62da1a4f-d60c-4cc6-b332-76e36ddb0bb0) 

## Features

- **Machine Learning Core**: Trained on 40,000+ articles using TF-IDF Vectorization & Logistic Regression.
- **REST API**: High-performance FastAPI backend for serving predictions.
- **Interactive UI**: A professional Streamlit dashboard with real-time confidence gauges.
- **Visual Analytics**: Breakdown of probability scores and confidence metrics.

## Tech Stack

- **Language**: Python 3.9+
- **ML**: Scikit-Learn, Pandas, NumPy
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Serialization**: Joblib

## Project Structure

```bash
fake-news-classifier/
│
├── app/                  # FastAPI Backend
│   ├── main.py           # API Endpoints
│   └── schema.py         # Pydantic Models
│
├── data/                 # Processed datasets
│
├── model/                # Serialized models (.pkl files)
│
├── training/             # Training scripts
│   └── train_model.py
│
├── ui/                   # Streamlit Frontend
│   └── streamlit_app.py
│
├── run.bat               # One-click startup script (Windows)
├── requirements.txt      # Dependencies
└── README.md             # Project Documentation

```

## Installation & Setup

1. **Clone the repository**
```bash
git clone [https://github.com/ToobaFazal02/fake-news-classifier.git](https://github.com/ToobaFazal02/fake-news-classifier.git)
cd fake-news-classifier

```


2. **Create & Activate Virtual Environment**
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

```


3. **Install Dependencies**
```bash
pip install -r requirements.txt

```



## How to Run

### Option 1: One-Click (Windows)

Double-click the `run.bat` file. It will automatically set up the environment and launch both servers.

### Option 2: Manual Start

**Step 1: Start Backend (API)**

```bash
uvicorn app.main:app --reload

```

*API runs at: http://127.0.0.1:8000*

**Step 2: Start Frontend (UI)**
Open a new terminal and run:

```bash
streamlit run ui/streamlit_app.py

```

## Model Performance

The Logistic Regression model was trained and evaluated on a balanced dataset.

| Metric | Score |
| --- | --- |
| **Accuracy** | **97%** |
| Precision | 0.96 |
| Recall | 0.97 |
| F1-Score | 0.97 |

```

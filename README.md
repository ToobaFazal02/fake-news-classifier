# Fake News Classifier

A full-stack machine learning application that detects whether a news article is **REAL** or **FAKE**. 

This project demonstrates an end-to-end ML pipeline: from data preprocessing and model training (Scikit-Learn) to serving predictions via a REST API (FastAPI) and an interactive frontend (Streamlit).

## Features

- **Machine Learning Core**: Uses TF-IDF Vectorization and Logistic Regression for high interpretability and efficiency.
- **REST API**: A robust FastAPI backend to serve predictions programmatically.
- **Interactive UI**: A Streamlit dashboard for real-time user testing.
- **Preprocessing Pipeline**: Includes custom text cleaning, tokenization, and stopword removal.

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
├── data/                 # Raw and processed datasets
│
├── model/                # Serialized models (.pkl files)
│
├── training/             # Training scripts and notebooks
│   └── train_model.py
│
├── ui/                   # Streamlit Frontend
│   └── streamlit_app.py
│
├── requirements.txt      # Dependencies
└── README.md             # Project Documentation

```

## Installation & Setup

1. **Clone the repository**
```bash
git clone [https://github.com/ToobaFazal02/fake-news-classifier.git](https://github.com/ToobaFazal02/fake-news-classifier.git)
cd fake-news-classifier

```


2. **Create a Virtual Environment**
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



## 🏃‍♂️ How to Run

### 1. Train the Model (First Run Only)

Before running the app, train the model to generate the `.pkl` files.

```bash
python training/train_model.py

```

### 2. Run the Backend (API)

Start the FastAPI server:

```bash
uvicorn app.main:app --reload

```

*API will be running at: `http://127.0.0.1:8000*`
*Swagger Docs: `http://127.0.0.1:8000/docs*`

### 3. Run the Frontend (UI)

In a new terminal, start the Streamlit app:

```bash
streamlit run ui/streamlit_app.py

```

## 📊 Model Performance

*(Metrics will be added here after training - e.g., Accuracy, Precision, Recall)*

## Contributing

Feel free to fork this repository and submit pull requests.



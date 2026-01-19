@echo off
echo ==============================================
echo 🚀 FAKE NEWS CLASSIFIER - STARTUP SCRIPT
echo ==============================================

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\Scripts\activate

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt --quiet

REM Download NLTK data
echo 📚 Downloading NLTK data...
python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('punkt', quiet=True); nltk.download('omw-1.4', quiet=True)"

REM Create necessary directories
if not exist "data" mkdir data
if not exist "model" mkdir model

REM Check if model exists
if not exist "model\classifier.pkl" (
    echo 🎓 Training model (first time setup)...
    python training/train_model.py
) else (
    echo ✅ Model already exists, skipping training
)

echo.
echo ==============================================
echo ✅ SETUP COMPLETE!
echo ==============================================
echo.
echo 🌐 Starting services...
echo.
echo 1️⃣  API: http://127.0.0.1:8000
echo 2️⃣  UI: http://localhost:8501
echo.
echo Press Ctrl+C to stop all services
echo.

REM Start API in new window
start "FastAPI Backend" cmd /k "call venv\Scripts\activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait for API to start
timeout /t 3 /nobreak > nul

REM Start Streamlit in new window
start "Streamlit UI" cmd /k "call venv\Scripts\activate && streamlit run ui/streamlit_app.py"

echo.
echo ✅ Both services started in separate windows!
echo.
pause
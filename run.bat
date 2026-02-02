@echo off
echo ========================================
echo IPL Auction Predictor - Starting Server
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from python.org
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "app.py" (
    echo ERROR: app.py not found
    echo Please make sure you're in the ipl-auction-predictor directory
    pause
    exit /b 1
)

REM Try to activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo WARNING: Virtual environment not found, using system Python
    echo It's recommended to run setup.bat first
    echo.
)

REM Check if model exists, train if needed
if not exist "model_artifacts\naive_bayes_model.pkl" (
    echo Model not found. Training model now...
    echo This will take a moment...
    echo.
    python model.py
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to train model
        echo Please check if all dependencies are installed
        echo Try running: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo.
    echo Model trained successfully!
    echo.
)

REM Check if frontend is built
if not exist "frontend\build\index.html" (
    echo WARNING: Frontend build not found
    echo The app will work but may not show the UI properly
    echo Run 'cd frontend && npm run build' to build frontend
    echo.
    pause
)

echo.
echo ========================================
echo Starting Flask Backend Server
echo ========================================
echo.
echo Server will be available at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python app.py

if errorlevel 1 (
    echo.
    echo ERROR: Server failed to start
    echo Common issues:
    echo - Port 5000 already in use
    echo - Missing dependencies (run: pip install -r requirements.txt)
    echo.
    pause
)

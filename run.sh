#!/bin/bash

echo "========================================"
echo "IPL Auction Predictor - Starting Server"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found"
    echo "Please run setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if model exists
if [ ! -f "model_artifacts/naive_bayes_model.pkl" ]; then
    echo "Model not found. Training model..."
    python model.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to train model"
        exit 1
    fi
fi

echo "Starting Flask backend server..."
echo ""
echo "Server will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""
python app.py

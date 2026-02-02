# Quick Start Guide 🚀

## For Windows 11 Users

### Prerequisites Check
- [ ] Python 3.8+ installed → Download from [python.org](https://www.python.org/downloads/)
- [ ] Node.js 16+ installed → Download from [nodejs.org](https://nodejs.org/)

### 3-Step Setup

#### Step 1: Run Setup Script
```cmd
setup.bat
```
This will:
- Create Python virtual environment
- Install all Python dependencies
- Install frontend dependencies
- Build the React frontend

#### Step 2: Start the Application
```cmd
run.bat
```

#### Step 3: Open Browser
Visit: **http://localhost:5000**

---

## For macOS/Linux Users

### Prerequisites Check
- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed

### 3-Step Setup

#### Step 1: Run Setup Script
```bash
chmod +x setup.sh
./setup.sh
```

#### Step 2: Start the Application
```bash
chmod +x run.sh
./run.sh
```

#### Step 3: Open Browser
Visit: **http://localhost:5000**

---

## Manual Setup (All Platforms)

If the automated scripts don't work:

```bash
# 1. Install Python dependencies
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

# 2. Install frontend dependencies
cd frontend
npm install
npm run build
cd ..

# 3. Run the application
python app.py
```

---

## Using the Application

### 1. Enter Player Details
- **Basic Info**: Age, Role, Country, Playing Style
- **Batting Stats**: Runs, Average, Strike Rate, etc.
- **Bowling Stats**: Wickets, Economy, Strike Rate, etc.
- **Performance**: Fitness, Consistency, Form, etc.

### 2. Get Prediction
- Click "PREDICT AUCTION PRICE"
- View predicted price in lakhs
- See confidence level
- Check price range

### 3. Sample Values (for testing)

**Example All-Rounder:**
- Age: 25
- Role: All-Rounder
- Country: India
- Domestic Matches: 100
- Runs: 4500, Average: 47, Strike Rate: 135
- Wickets: 85, Economy: 7.2, Strike Rate: 22
- Fitness: 92, Consistency: 88

---

## Common Issues

### Issue: "Python not found"
**Solution**: Install Python from python.org and add to PATH

### Issue: "Node not found"
**Solution**: Install Node.js from nodejs.org

### Issue: "Port 5000 already in use"
**Solution**: 
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:5000 | xargs kill -9
```

### Issue: "Module not found"
**Solution**: Activate virtual environment and reinstall
```bash
pip install -r requirements.txt
```

---

## File Structure

```
ipl-auction-predictor/
├── app.py                  # Backend server
├── model.py               # ML model
├── players_dataset.csv    # 50K players dataset
├── requirements.txt       # Python packages
├── setup.bat / setup.sh   # Setup scripts
├── run.bat / run.sh       # Run scripts
├── frontend/              # React frontend
└── model_artifacts/       # Trained model files
```

---

## Next Steps

1. ✅ Run the application locally
2. 📊 Try different player profiles
3. 🔧 Customize parameters
4. 🌐 Deploy to AWS (see AWS_DEPLOYMENT.md)

---

## Need Help?

1. Check **README.md** for detailed documentation
2. Check **AWS_DEPLOYMENT.md** for deployment guide
3. Check terminal/console for error messages

---

**Enjoy predicting IPL auction prices! 🏏**

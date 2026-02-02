# TROUBLESHOOTING GUIDE - Windows 11

## If run.bat Crashes or Doesn't Work

### Method 1: Use run-simple.bat (EASIEST)
Just double-click `run-simple.bat` - this is the simplest way to run the app.

---

### Method 2: Manual Steps (MOST RELIABLE)

Open **Command Prompt** and run these commands one by one:

```cmd
cd path\to\ipl-auction-predictor

pip install flask flask-cors pandas numpy scikit-learn

python app.py
```

Then open browser: `http://localhost:5000`

---

### Method 3: Step-by-Step Setup

#### Step 1: Install Dependencies
```cmd
cd path\to\ipl-auction-predictor
pip install -r requirements.txt
```

#### Step 2: Build Frontend (One time only)
```cmd
cd frontend
npm install
npm run build
cd ..
```

#### Step 3: Run Server
```cmd
python app.py
```

---

## Common Issues & Solutions

### ❌ Issue: "Python is not recognized"
**Solution:**
1. Install Python from [python.org](https://www.python.org/downloads/)
2. During installation, CHECK "Add Python to PATH"
3. Restart Command Prompt

### ❌ Issue: "Port 5000 already in use"
**Solution 1 - Kill process:**
```cmd
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F
```

**Solution 2 - Use different port:**
Edit `app.py`, change last line:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Changed to 5001
```
Then visit: `http://localhost:5001`

### ❌ Issue: "Module not found"
**Solution:**
```cmd
pip install flask flask-cors pandas numpy scikit-learn
```

### ❌ Issue: "Virtual environment error"
**Solution:**
Don't use virtual environment, just run directly:
```cmd
python app.py
```

### ❌ Issue: "Frontend not showing"
**Solution:**
```cmd
cd frontend
npm install
npm run build
cd ..
python app.py
```

### ❌ Issue: "npm not recognized"
**Solution:**
1. Install Node.js from [nodejs.org](https://nodejs.org/)
2. Restart Command Prompt
3. Run `npm install` again

---

## Quick Start (No Setup Required)

If you just want to test the app quickly:

1. Make sure Python is installed
2. Open Command Prompt
3. Navigate to project folder:
   ```cmd
   cd C:\path\to\ipl-auction-predictor
   ```
4. Install packages:
   ```cmd
   pip install flask flask-cors pandas numpy scikit-learn
   ```
5. Run:
   ```cmd
   python app.py
   ```
6. Open browser: `http://localhost:5000`

---

## Alternative: Python Interactive Mode

If nothing else works, you can run it manually:

1. Open Command Prompt
2. Navigate to folder:
   ```cmd
   cd C:\path\to\ipl-auction-predictor
   ```
3. Start Python:
   ```cmd
   python
   ```
4. In Python console:
   ```python
   import subprocess
   subprocess.run(["pip", "install", "flask", "flask-cors", "pandas", "numpy", "scikit-learn"])
   exit()
   ```
5. Then run:
   ```cmd
   python app.py
   ```

---

## Check if Everything is Installed

Run these commands to verify:

```cmd
python --version
# Should show Python 3.8 or higher

pip --version
# Should show pip version

python -c "import flask; print('Flask OK')"
# Should print "Flask OK"

python -c "import pandas; print('Pandas OK')"
# Should print "Pandas OK"
```

---

## Still Not Working?

### Last Resort - Direct Python Execution:

1. Open `app.py` in a text editor
2. Check the last line says: `app.run(debug=True, host='0.0.0.0', port=5000)`
3. Open Command Prompt in the project folder
4. Run:
   ```cmd
   python -m flask run --host=0.0.0.0 --port=5000
   ```

---

## Frontend Issues

If you see the app but it looks broken:

```cmd
cd frontend
del /s /q build
del /s /q node_modules
npm install
npm run build
cd ..
python app.py
```

---

## Permission Issues

If you get "Access Denied" errors:

1. Right-click Command Prompt
2. Select "Run as Administrator"
3. Try again

---

## What to Check:

✅ Python installed (3.8 or higher)
✅ pip working
✅ In correct directory (has app.py)
✅ Dependencies installed (requirements.txt)
✅ Port 5000 is free
✅ No antivirus blocking Python

---

## Need More Help?

If you're still stuck, tell me:
1. What exact error message you see
2. What command you ran
3. Your Python version (`python --version`)
4. Whether you ran setup.bat first

I'll help you fix it! 😊

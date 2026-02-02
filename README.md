# IPL Auction Price Predictor 🏏

A comprehensive machine learning application that predicts IPL auction prices for debuting players based on their domestic cricket performance using Naive Bayes algorithm.

## 📊 Project Overview

- **Dataset**: 50,000 synthetic player records with 32 parameters
- **Algorithm**: Gaussian Naive Bayes Classification
- **Frontend**: React.js with Framer Motion animations
- **Backend**: Flask REST API
- **Features**: 
  - 20+ performance parameters
  - Real-time price prediction
  - Confidence scoring
  - IPL-themed responsive UI
  - Interactive data visualization

## 🎯 Features

### Player Parameters (32 Total)
1. **Basic Information**: Age, Role, Country, Batting Style, Bowling Style
2. **Experience**: Domestic Matches, Experience Factor
3. **Batting Stats**: Innings, Runs, Average, Strike Rate, Hundreds, Fifties, Highest Score, Boundary %
4. **Bowling Stats**: Overs, Wickets, Bowling Average, Economy Rate, Strike Rate, 5-wicket hauls, Best Bowling, Dot Ball %
5. **Fielding**: Catches, Stumpings
6. **Performance Metrics**: Consistency Rating, Fitness Score, Recent Form, Match-Winning Performances, Pressure Handling

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn

### Windows 11 Setup

#### Step 1: Install Python Dependencies
```bash
# Navigate to project directory
cd ipl-auction-predictor

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt
```

#### Step 2: Train the Model
```bash
# The model should already be trained, but if you need to retrain:
python model.py
```

#### Step 3: Install Frontend Dependencies
```bash
# Navigate to frontend directory
cd frontend

# Install npm packages
npm install

# Build the frontend
npm run build

# Go back to root directory
cd ..
```

#### Step 4: Run the Application

**Option 1: Development Mode**
```bash
# Terminal 1 - Run Backend
python app.py

# Terminal 2 - Run Frontend (in frontend directory)
cd frontend
npm start
```

**Option 2: Production Mode**
```bash
# Run backend (serves built frontend)
python app.py

# Access at: http://localhost:5000
```

### macOS/Linux Setup

```bash
# Navigate to project directory
cd ipl-auction-predictor

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
npm run build
cd ..

# Run application
python app.py
```

## 📁 Project Structure

```
ipl-auction-predictor/
├── app.py                      # Flask backend server
├── model.py                    # Naive Bayes model implementation
├── players_dataset.csv         # Generated dataset (50,000 players)
├── requirements.txt            # Python dependencies
├── model_artifacts/            # Trained model files
│   ├── naive_bayes_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── price_bins.pkl
│   └── feature_columns.pkl
└── frontend/                   # React frontend
    ├── package.json
    ├── public/
    │   └── index.html
    ├── src/
    │   ├── App.js
    │   ├── App.css
    │   ├── index.js
    │   └── index.css
    └── build/                  # Production build (after npm run build)
```

## 🔌 API Endpoints

### Health Check
```http
GET /api/health
Response: { "status": "healthy", "model_loaded": true }
```

### Predict Price
```http
POST /api/predict
Content-Type: application/json

Request Body: {
  "age": 25,
  "role": "All-Rounder",
  "country": "India",
  ... (all 30 parameters)
}

Response: {
  "success": true,
  "prediction": {
    "predicted_price": 450.50,
    "confidence": 78.23,
    "price_range": {
      "min": 380.00,
      "max": 520.00
    }
  }
}
```

### Dataset Statistics
```http
GET /api/dataset-stats
Response: {
  "success": true,
  "stats": {
    "total_players": 50000,
    "avg_price": 316.22,
    "max_price": 2000.00,
    ...
  }
}
```

### Sample Players
```http
GET /api/sample-players
Response: { "success": true, "players": [...] }
```

## 🎨 UI Features

- **IPL-Themed Design**: Blue, red, and gold color scheme
- **Smooth Animations**: Framer Motion powered transitions
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Tabbed Interface**: Organized input for batting/bowling/fielding stats
- **Real-time Validation**: Form validation and error handling
- **Confidence Indicator**: Visual confidence meter for predictions
- **Dataset Stats**: Live statistics dashboard

## 📊 Model Performance

- **Algorithm**: Gaussian Naive Bayes (sklearn)
- **Training Data**: 40,000 players (80% split)
- **Test Data**: 10,000 players (20% split)
- **Metrics**:
  - MAE: ~167 lakhs
  - RMSE: ~324 lakhs
  - R² Score: ~0.29

## 🌐 Deployment on AWS

### AWS Deployment Steps

1. **EC2 Instance Setup**
```bash
# Launch Ubuntu 22.04 EC2 instance
# Security Group: Allow ports 22 (SSH), 80 (HTTP), 443 (HTTPS), 5000

# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3-pip python3-venv -y

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install nodejs -y
```

2. **Deploy Application**
```bash
# Clone/upload project to EC2
# Or use git clone if you have it in a repository

# Setup Python
cd ipl-auction-predictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Build frontend
cd frontend
npm install
npm run build
cd ..

# Run with production server (Gunicorn)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

3. **Setup Nginx (Optional)**
```bash
# Install Nginx
sudo apt install nginx -y

# Configure Nginx
sudo nano /etc/nginx/sites-available/ipl-predictor

# Add configuration:
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/ipl-predictor /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

4. **Setup Systemd Service**
```bash
sudo nano /etc/systemd/system/ipl-predictor.service

# Add:
[Unit]
Description=IPL Auction Predictor
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/ipl-auction-predictor
Environment="PATH=/home/ubuntu/ipl-auction-predictor/venv/bin"
ExecStart=/home/ubuntu/ipl-auction-predictor/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable ipl-predictor
sudo systemctl start ipl-predictor
```

## 🛠️ Development

### Adding New Features
1. Update model parameters in `model.py`
2. Retrain model: `python model.py`
3. Update frontend form in `App.js`
4. Update API endpoints in `app.py`

### Debugging
```bash
# Check backend logs
python app.py  # View console output

# Check frontend in development mode
cd frontend
npm start  # Visit http://localhost:3000

# Test API directly
curl http://localhost:5000/api/health
```

## 📝 Usage Example

1. **Start the application**:
   - Run `python app.py`
   - Visit `http://localhost:5000`

2. **Enter player statistics**:
   - Fill in basic information (age, role, country)
   - Add batting statistics (runs, average, strike rate)
   - Add bowling statistics (wickets, economy, strike rate)
   - Add fielding and performance metrics

3. **Get prediction**:
   - Click "PREDICT AUCTION PRICE"
   - View predicted price with confidence level
   - See price range estimation

## 🤝 Contributing

This is a demonstration project. Feel free to fork and enhance!

## 📄 License

This project is for educational purposes.

## 👨‍💻 Technical Stack

- **Machine Learning**: scikit-learn (Naive Bayes)
- **Backend**: Flask, Python 3.8+
- **Frontend**: React 18, Framer Motion
- **Data Processing**: pandas, NumPy
- **Styling**: Custom CSS with CSS Variables
- **Fonts**: Rajdhani, Bebas Neue (Google Fonts)

## 🎯 Future Enhancements

- [ ] Add more ML algorithms for comparison
- [ ] Real IPL player data integration
- [ ] Historical auction data analysis
- [ ] Team budget constraints simulation
- [ ] Player comparison feature
- [ ] Export predictions to PDF
- [ ] User authentication
- [ ] Database integration

## 📞 Support

For issues or questions:
1. Check the README
2. Review API documentation
3. Check browser console for frontend errors
4. Check terminal for backend errors

---

**Built with ❤️ for Cricket Analytics**

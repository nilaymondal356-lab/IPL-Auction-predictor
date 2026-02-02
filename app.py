from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from model import IPLAuctionPredictor
import os
import pandas as pd

app = Flask(__name__, static_folder='frontend/build', static_url_path='')
CORS(app)

# Initialize predictor
predictor = IPLAuctionPredictor()

# Load trained model
try:
    predictor.load_model('model_artifacts')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please train the model first by running: python model.py")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.is_trained
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict auction price for a player"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = [
            'age', 'role', 'country', 'batting_style', 'bowling_style',
            'domestic_matches', 'innings_batted', 'runs_scored',
            'batting_average', 'batting_strike_rate', 'hundreds', 'fifties',
            'highest_score', 'boundary_percentage', 'overs_bowled',
            'wickets_taken', 'bowling_average', 'economy_rate',
            'bowling_strike_rate', 'five_wicket_hauls', 'best_bowling_wickets',
            'dot_ball_percentage', 'catches', 'stumpings',
            'consistency_rating', 'fitness_score', 'experience_factor',
            'recent_form_rating', 'match_winning_performances',
            'pressure_handling_score'
        ]
        
        # Convert numeric fields
        numeric_fields = [f for f in required_fields if f not in ['role', 'country', 'batting_style', 'bowling_style']]
        for field in numeric_fields:
            if field in data:
                data[field] = float(data[field])
        
        # Make prediction
        result = predictor.predict(data)
        
        return jsonify({
            'success': True,
            'prediction': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/dataset-stats', methods=['GET'])
def dataset_stats():
    """Get dataset statistics"""
    try:
        df = pd.read_csv('players_dataset.csv')
        
        stats = {
            'total_players': len(df),
            'avg_price': round(df['auction_price_lakhs'].mean(), 2),
            'max_price': round(df['auction_price_lakhs'].max(), 2),
            'min_price': round(df['auction_price_lakhs'].min(), 2),
            'role_distribution': df['role'].value_counts().to_dict(),
            'country_distribution': df['country'].value_counts().to_dict(),
            'avg_age': round(df['age'].mean(), 2),
            'total_runs': int(df['runs_scored'].sum()),
            'total_wickets': int(df['wickets_taken'].sum())
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/sample-players', methods=['GET'])
def sample_players():
    """Get sample players for reference"""
    try:
        df = pd.read_csv('players_dataset.csv')
        
        # Get diverse samples
        samples = []
        for role in df['role'].unique():
            role_df = df[df['role'] == role]
            sample = role_df.sample(min(2, len(role_df)))
            samples.append(sample)
        
        result_df = pd.concat(samples)
        
        return jsonify({
            'success': True,
            'players': result_df.to_dict('records')
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Serve React frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Run on localhost
    print("="*60)
    print("IPL Auction Price Predictor - Backend Server")
    print("="*60)
    print("Server running on: http://localhost:5000")
    print("API Endpoints:")
    print("  - GET  /api/health         : Health check")
    print("  - POST /api/predict        : Predict player price")
    print("  - GET  /api/dataset-stats  : Get dataset statistics")
    print("  - GET  /api/sample-players : Get sample players")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

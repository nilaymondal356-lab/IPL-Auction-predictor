import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pickle
import os

class IPLAuctionPredictor:
    def __init__(self):
        self.model = GaussianNB()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.price_bins = []
        self.is_trained = False
        
    def create_price_bins(self, prices, n_bins=20):
        """Create price bins for classification"""
        self.price_bins = np.percentile(prices, np.linspace(0, 100, n_bins + 1))
        return np.digitize(prices, self.price_bins[1:-1])
    
    def price_from_bin(self, bin_idx):
        """Convert bin index back to price"""
        if bin_idx < len(self.price_bins) - 1:
            return (self.price_bins[bin_idx] + self.price_bins[bin_idx + 1]) / 2
        else:
            return self.price_bins[-1]
    
    def preprocess_data(self, df, fit=False):
        """Preprocess the dataset"""
        df = df.copy()
        
        # Categorical columns to encode
        categorical_cols = ['role', 'country', 'batting_style', 'bowling_style']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Select numeric features
        feature_cols = [
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
        
        # Filter existing columns
        self.feature_columns = [col for col in feature_cols if col in df.columns]
        
        X = df[self.feature_columns].fillna(0)
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def train(self, csv_path):
        """Train the Naive Bayes model"""
        print("Loading dataset...")
        df = pd.read_csv(csv_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {len(self.feature_columns)}")
        
        # Preprocess features
        X = self.preprocess_data(df, fit=True)
        
        # Create price bins for classification
        y = self.create_price_bins(df['auction_price_lakhs'].values)
        
        print(f"Number of price bins: {len(self.price_bins)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Training Naive Bayes model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        # Convert bins back to prices for evaluation
        y_test_prices = np.array([self.price_from_bin(i) for i in y_test])
        y_pred_prices = np.array([self.price_from_bin(i) for i in y_pred])
        
        mae = mean_absolute_error(y_test_prices, y_pred_prices)
        rmse = np.sqrt(mean_squared_error(y_test_prices, y_pred_prices))
        r2 = r2_score(y_test_prices, y_pred_prices)
        
        print(f"\nModel Performance:")
        print(f"MAE: ₹{mae:.2f} lakhs")
        print(f"RMSE: ₹{rmse:.2f} lakhs")
        print(f"R² Score: {r2:.4f}")
        
        self.is_trained = True
        return mae, rmse, r2
    
    def predict(self, player_data):
        """Predict auction price for a player"""
        if not self.is_trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Convert dict to DataFrame
        df = pd.DataFrame([player_data])
        
        # Preprocess
        X = self.preprocess_data(df, fit=False)
        
        # Predict bin
        bin_pred = self.model.predict(X)[0]
        
        # Convert to price
        price = self.price_from_bin(bin_pred)
        
        # Get probability distribution
        proba = self.model.predict_proba(X)[0]
        confidence = max(proba) * 100
        
        return {
            'predicted_price': round(price, 2),
            'confidence': round(confidence, 2),
            'price_range': {
                'min': round(self.price_bins[max(0, bin_pred - 1)], 2),
                'max': round(self.price_bins[min(len(self.price_bins) - 1, bin_pred + 1)], 2)
            }
        }
    
    def save_model(self, path='model_artifacts'):
        """Save trained model and preprocessors"""
        os.makedirs(path, exist_ok=True)
        
        with open(f'{path}/naive_bayes_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(f'{path}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f'{path}/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        with open(f'{path}/price_bins.pkl', 'wb') as f:
            pickle.dump(self.price_bins, f)
        
        with open(f'{path}/feature_columns.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        print(f"Model saved to {path}/")
    
    def load_model(self, path='model_artifacts'):
        """Load trained model and preprocessors"""
        with open(f'{path}/naive_bayes_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        with open(f'{path}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(f'{path}/label_encoders.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        with open(f'{path}/price_bins.pkl', 'rb') as f:
            self.price_bins = pickle.load(f)
        
        with open(f'{path}/feature_columns.pkl', 'rb') as f:
            self.feature_columns = pickle.load(f)
        
        self.is_trained = True
        print(f"Model loaded from {path}/")


if __name__ == "__main__":
    # Train and save model
    predictor = IPLAuctionPredictor()
    predictor.train('players_dataset.csv')
    predictor.save_model()
    
    # Test prediction
    test_player = {
        'age': 25,
        'role': 'All-Rounder',
        'country': 'India',
        'batting_style': 'Right-Hand',
        'bowling_style': 'Right-Arm Fast',
        'domestic_matches': 100,
        'innings_batted': 95,
        'runs_scored': 4500,
        'batting_average': 47.37,
        'batting_strike_rate': 135.5,
        'hundreds': 8,
        'fifties': 22,
        'highest_score': 156,
        'boundary_percentage': 22.5,
        'overs_bowled': 450.5,
        'wickets_taken': 85,
        'bowling_average': 26.5,
        'economy_rate': 7.2,
        'bowling_strike_rate': 22.1,
        'five_wicket_hauls': 3,
        'best_bowling_wickets': 5,
        'dot_ball_percentage': 45.2,
        'catches': 55,
        'stumpings': 0,
        'consistency_rating': 88.5,
        'fitness_score': 92.0,
        'experience_factor': 85.0,
        'recent_form_rating': 89.5,
        'match_winning_performances': 18,
        'pressure_handling_score': 87.5
    }
    
    result = predictor.predict(test_player)
    print("\n" + "="*50)
    print("TEST PREDICTION:")
    print("="*50)
    print(f"Predicted Price: ₹{result['predicted_price']} lakhs")
    print(f"Confidence: {result['confidence']}%")
    print(f"Price Range: ₹{result['price_range']['min']} - ₹{result['price_range']['max']} lakhs")

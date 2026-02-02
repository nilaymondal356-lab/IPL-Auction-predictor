import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    age: '',
    role: 'Batsman',
    country: 'India',
    batting_style: 'Right-Hand',
    bowling_style: 'Right-Arm Fast',
    domestic_matches: '',
    innings_batted: '',
    runs_scored: '',
    batting_average: '',
    batting_strike_rate: '',
    hundreds: '',
    fifties: '',
    highest_score: '',
    boundary_percentage: '',
    overs_bowled: '',
    wickets_taken: '',
    bowling_average: '',
    economy_rate: '',
    bowling_strike_rate: '',
    five_wicket_hauls: '',
    best_bowling_wickets: '',
    dot_ball_percentage: '',
    catches: '',
    stumpings: '',
    consistency_rating: '',
    fitness_score: '',
    experience_factor: '',
    recent_form_rating: '',
    match_winning_performances: '',
    pressure_handling_score: ''
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [activeTab, setActiveTab] = useState('batting');
  const [validationErrors, setValidationErrors] = useState({});

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get('/api/dataset-stats');
      if (response.data.success) {
        setStats(response.data.stats);
      }
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: value
    }));
  };

  const validateForm = () => {
    const errors = {};
    const requiredNumericFields = [
      'age', 'domestic_matches', 'innings_batted', 'runs_scored',
      'batting_average', 'batting_strike_rate', 'hundreds', 'fifties',
      'highest_score', 'boundary_percentage', 'overs_bowled', 'wickets_taken',
      'bowling_average', 'economy_rate', 'bowling_strike_rate',
      'five_wicket_hauls', 'best_bowling_wickets', 'dot_ball_percentage',
      'catches', 'stumpings', 'consistency_rating', 'fitness_score',
      'experience_factor', 'recent_form_rating', 'match_winning_performances',
      'pressure_handling_score'
    ];

    requiredNumericFields.forEach(field => {
      if (!formData[field] || formData[field] === '') {
        errors[field] = 'Required';
      }
    });

    return errors;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const errors = validateForm();
    
    if (Object.keys(errors).length > 0) {
      setValidationErrors(errors);
      alert(`⚠️ Please fill all required fields! ${Object.keys(errors).length} field(s) are missing.`);
      return;
    }

    setValidationErrors({});
    setLoading(true);
    setPrediction(null);

    try {
      const response = await axios.post('/api/predict', formData);
      if (response.data.success) {
        setPrediction(response.data.prediction);
        setTimeout(() => {
          const predictionElement = document.querySelector('.prediction-container');
          if (predictionElement) {
            predictionElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        }, 100);
      }
    } catch (error) {
      const errorMessage = error.response?.data?.error || error.message;
      alert('❌ Error: ' + errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const convertToCrores = (lakhs) => {
    const crores = lakhs / 100;
    return crores.toFixed(2);
  };

  const getPlayerTier = (crores) => {
    const price = parseFloat(crores);
    if (price < 20) {
      return {
        tier: 'EMERGING',
        color: '#10b981',
        bgColor: 'rgba(16, 185, 129, 0.1)',
        description: 'Rising Talent'
      };
    } else if (price < 48) {
      return {
        tier: 'ESTABLISHED',
        color: '#f59e0b',
        bgColor: 'rgba(245, 158, 11, 0.1)',
        description: 'Proven Performer'
      };
    } else {
      return {
        tier: 'ELITE',
        color: '#ef4444',
        bgColor: 'rgba(239, 68, 68, 0.1)',
        description: 'Premium Player'
      };
    }
  };

  const roles = ['Batsman', 'Bowler', 'All-Rounder', 'Wicket-Keeper'];
  const countries = ['India', 'Australia', 'England', 'South Africa', 'New Zealand', 'West Indies', 'Pakistan', 'Sri Lanka', 'Bangladesh', 'Afghanistan'];
  const battingStyles = ['Right-Hand', 'Left-Hand'];
  const bowlingStyles = ['Right-Arm Fast', 'Left-Arm Fast', 'Right-Arm Medium', 'Left-Arm Medium', 'Right-Arm Spin', 'Left-Arm Spin', 'Leg-Spin', 'Off-Spin'];

  return (
    <div className="app">
      <div className="bg-pattern"></div>
      
      <motion.header 
        className="header"
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        <div className="header-content">
          <motion.h1 
            className="title"
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
          >
            IPL AUCTION PREDICTOR
          </motion.h1>
          <motion.p 
            className="subtitle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            Powered by Naive Bayes Algorithm
          </motion.p>
        </div>
      </motion.header>

      {stats && (
        <motion.div 
          className="stats-bar"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
        >
          <div className="stat-item">
            <span className="stat-label">Total Players</span>
            <span className="stat-value">{stats.total_players.toLocaleString()}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Avg Price</span>
            <span className="stat-value">₹{convertToCrores(stats.avg_price)} Cr</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Max Price</span>
            <span className="stat-value">₹{convertToCrores(stats.max_price)} Cr</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Avg Age</span>
            <span className="stat-value">{stats.avg_age}</span>
          </div>
        </motion.div>
      )}

      <div className="main-content">
        <motion.div 
          className="form-container"
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3, duration: 0.6 }}
        >
          <form onSubmit={handleSubmit}>
            <div className="form-section">
              <h2 className="section-title">Basic Information</h2>
              <div className="form-grid">
                <div className="form-group">
                  <label>Age <span className="required">*</span></label>
                  <input
                    type="number"
                    name="age"
                    value={formData.age}
                    onChange={handleChange}
                    min="18"
                    max="40"
                    className={validationErrors.age ? 'error' : ''}
                  />
                  {validationErrors.age && <span className="error-message">{validationErrors.age}</span>}
                </div>
                <div className="form-group">
                  <label>Role <span className="required">*</span></label>
                  <select name="role" value={formData.role} onChange={handleChange}>
                    {roles.map(role => <option key={role} value={role}>{role}</option>)}
                  </select>
                </div>
                <div className="form-group">
                  <label>Country <span className="required">*</span></label>
                  <select name="country" value={formData.country} onChange={handleChange}>
                    {countries.map(country => <option key={country} value={country}>{country}</option>)}
                  </select>
                </div>
                <div className="form-group">
                  <label>Batting Style <span className="required">*</span></label>
                  <select name="batting_style" value={formData.batting_style} onChange={handleChange}>
                    {battingStyles.map(style => <option key={style} value={style}>{style}</option>)}
                  </select>
                </div>
                <div className="form-group">
                  <label>Bowling Style <span className="required">*</span></label>
                  <select name="bowling_style" value={formData.bowling_style} onChange={handleChange}>
                    {bowlingStyles.map(style => <option key={style} value={style}>{style}</option>)}
                  </select>
                </div>
                <div className="form-group">
                  <label>Domestic Matches <span className="required">*</span></label>
                  <input
                    type="number"
                    name="domestic_matches"
                    value={formData.domestic_matches}
                    onChange={handleChange}
                    min="0"
                    className={validationErrors.domestic_matches ? 'error' : ''}
                  />
                  {validationErrors.domestic_matches && <span className="error-message">{validationErrors.domestic_matches}</span>}
                </div>
              </div>
            </div>

            <div className="tabs">
              <button
                type="button"
                className={`tab ${activeTab === 'batting' ? 'active' : ''}`}
                onClick={() => setActiveTab('batting')}
              >
                Batting Stats
              </button>
              <button
                type="button"
                className={`tab ${activeTab === 'bowling' ? 'active' : ''}`}
                onClick={() => setActiveTab('bowling')}
              >
                Bowling Stats
              </button>
              <button
                type="button"
                className={`tab ${activeTab === 'fielding' ? 'active' : ''}`}
                onClick={() => setActiveTab('fielding')}
              >
                Fielding & Performance
              </button>
            </div>

            <AnimatePresence mode="wait">
              {activeTab === 'batting' && (
                <motion.div
                  key="batting"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                  className="form-section"
                >
                  <h2 className="section-title">Batting Statistics</h2>
                  <div className="form-grid">
                    <div className="form-group">
                      <label>Innings Batted <span className="required">*</span></label>
                      <input type="number" name="innings_batted" value={formData.innings_batted} onChange={handleChange} min="0" className={validationErrors.innings_batted ? 'error' : ''} />
                      {validationErrors.innings_batted && <span className="error-message">{validationErrors.innings_batted}</span>}
                    </div>
                    <div className="form-group">
                      <label>Runs Scored <span className="required">*</span></label>
                      <input type="number" name="runs_scored" value={formData.runs_scored} onChange={handleChange} min="0" className={validationErrors.runs_scored ? 'error' : ''} />
                      {validationErrors.runs_scored && <span className="error-message">{validationErrors.runs_scored}</span>}
                    </div>
                    <div className="form-group">
                      <label>Batting Average <span className="required">*</span></label>
                      <input type="number" step="0.01" name="batting_average" value={formData.batting_average} onChange={handleChange} min="0" className={validationErrors.batting_average ? 'error' : ''} />
                      {validationErrors.batting_average && <span className="error-message">{validationErrors.batting_average}</span>}
                    </div>
                    <div className="form-group">
                      <label>Strike Rate <span className="required">*</span></label>
                      <input type="number" step="0.01" name="batting_strike_rate" value={formData.batting_strike_rate} onChange={handleChange} min="0" className={validationErrors.batting_strike_rate ? 'error' : ''} />
                      {validationErrors.batting_strike_rate && <span className="error-message">{validationErrors.batting_strike_rate}</span>}
                    </div>
                    <div className="form-group">
                      <label>Hundreds <span className="required">*</span></label>
                      <input type="number" name="hundreds" value={formData.hundreds} onChange={handleChange} min="0" className={validationErrors.hundreds ? 'error' : ''} />
                      {validationErrors.hundreds && <span className="error-message">{validationErrors.hundreds}</span>}
                    </div>
                    <div className="form-group">
                      <label>Fifties <span className="required">*</span></label>
                      <input type="number" name="fifties" value={formData.fifties} onChange={handleChange} min="0" className={validationErrors.fifties ? 'error' : ''} />
                      {validationErrors.fifties && <span className="error-message">{validationErrors.fifties}</span>}
                    </div>
                    <div className="form-group">
                      <label>Highest Score <span className="required">*</span></label>
                      <input type="number" name="highest_score" value={formData.highest_score} onChange={handleChange} min="0" className={validationErrors.highest_score ? 'error' : ''} />
                      {validationErrors.highest_score && <span className="error-message">{validationErrors.highest_score}</span>}
                    </div>
                    <div className="form-group">
                      <label>Boundary % <span className="required">*</span></label>
                      <input type="number" step="0.01" name="boundary_percentage" value={formData.boundary_percentage} onChange={handleChange} min="0" max="100" className={validationErrors.boundary_percentage ? 'error' : ''} />
                      {validationErrors.boundary_percentage && <span className="error-message">{validationErrors.boundary_percentage}</span>}
                    </div>
                  </div>
                </motion.div>
              )}

              {activeTab === 'bowling' && (
                <motion.div
                  key="bowling"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                  className="form-section"
                >
                  <h2 className="section-title">Bowling Statistics</h2>
                  <div className="form-grid">
                    <div className="form-group">
                      <label>Overs Bowled <span className="required">*</span></label>
                      <input type="number" step="0.1" name="overs_bowled" value={formData.overs_bowled} onChange={handleChange} min="0" className={validationErrors.overs_bowled ? 'error' : ''} />
                      {validationErrors.overs_bowled && <span className="error-message">{validationErrors.overs_bowled}</span>}
                    </div>
                    <div className="form-group">
                      <label>Wickets Taken <span className="required">*</span></label>
                      <input type="number" name="wickets_taken" value={formData.wickets_taken} onChange={handleChange} min="0" className={validationErrors.wickets_taken ? 'error' : ''} />
                      {validationErrors.wickets_taken && <span className="error-message">{validationErrors.wickets_taken}</span>}
                    </div>
                    <div className="form-group">
                      <label>Bowling Average <span className="required">*</span></label>
                      <input type="number" step="0.01" name="bowling_average" value={formData.bowling_average} onChange={handleChange} min="0" className={validationErrors.bowling_average ? 'error' : ''} />
                      {validationErrors.bowling_average && <span className="error-message">{validationErrors.bowling_average}</span>}
                    </div>
                    <div className="form-group">
                      <label>Economy Rate <span className="required">*</span></label>
                      <input type="number" step="0.01" name="economy_rate" value={formData.economy_rate} onChange={handleChange} min="0" className={validationErrors.economy_rate ? 'error' : ''} />
                      {validationErrors.economy_rate && <span className="error-message">{validationErrors.economy_rate}</span>}
                    </div>
                    <div className="form-group">
                      <label>Bowling Strike Rate <span className="required">*</span></label>
                      <input type="number" step="0.01" name="bowling_strike_rate" value={formData.bowling_strike_rate} onChange={handleChange} min="0" className={validationErrors.bowling_strike_rate ? 'error' : ''} />
                      {validationErrors.bowling_strike_rate && <span className="error-message">{validationErrors.bowling_strike_rate}</span>}
                    </div>
                    <div className="form-group">
                      <label>5-Wicket Hauls <span className="required">*</span></label>
                      <input type="number" name="five_wicket_hauls" value={formData.five_wicket_hauls} onChange={handleChange} min="0" className={validationErrors.five_wicket_hauls ? 'error' : ''} />
                      {validationErrors.five_wicket_hauls && <span className="error-message">{validationErrors.five_wicket_hauls}</span>}
                    </div>
                    <div className="form-group">
                      <label>Best Bowling <span className="required">*</span></label>
                      <input type="number" name="best_bowling_wickets" value={formData.best_bowling_wickets} onChange={handleChange} min="0" max="10" className={validationErrors.best_bowling_wickets ? 'error' : ''} />
                      {validationErrors.best_bowling_wickets && <span className="error-message">{validationErrors.best_bowling_wickets}</span>}
                    </div>
                    <div className="form-group">
                      <label>Dot Ball % <span className="required">*</span></label>
                      <input type="number" step="0.01" name="dot_ball_percentage" value={formData.dot_ball_percentage} onChange={handleChange} min="0" max="100" className={validationErrors.dot_ball_percentage ? 'error' : ''} />
                      {validationErrors.dot_ball_percentage && <span className="error-message">{validationErrors.dot_ball_percentage}</span>}
                    </div>
                  </div>
                </motion.div>
              )}

              {activeTab === 'fielding' && (
                <motion.div
                  key="fielding"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                  className="form-section"
                >
                  <h2 className="section-title">Fielding & Performance Metrics</h2>
                  <div className="form-grid">
                    <div className="form-group">
                      <label>Catches <span className="required">*</span></label>
                      <input type="number" name="catches" value={formData.catches} onChange={handleChange} min="0" className={validationErrors.catches ? 'error' : ''} />
                      {validationErrors.catches && <span className="error-message">{validationErrors.catches}</span>}
                    </div>
                    <div className="form-group">
                      <label>Stumpings <span className="required">*</span></label>
                      <input type="number" name="stumpings" value={formData.stumpings} onChange={handleChange} min="0" className={validationErrors.stumpings ? 'error' : ''} />
                      {validationErrors.stumpings && <span className="error-message">{validationErrors.stumpings}</span>}
                    </div>
                    <div className="form-group">
                      <label>Consistency Rating (0-100) <span className="required">*</span></label>
                      <input type="number" step="0.01" name="consistency_rating" value={formData.consistency_rating} onChange={handleChange} min="0" max="100" className={validationErrors.consistency_rating ? 'error' : ''} />
                      {validationErrors.consistency_rating && <span className="error-message">{validationErrors.consistency_rating}</span>}
                    </div>
                    <div className="form-group">
                      <label>Fitness Score (0-100) <span className="required">*</span></label>
                      <input type="number" step="0.01" name="fitness_score" value={formData.fitness_score} onChange={handleChange} min="0" max="100" className={validationErrors.fitness_score ? 'error' : ''} />
                      {validationErrors.fitness_score && <span className="error-message">{validationErrors.fitness_score}</span>}
                    </div>
                    <div className="form-group">
                      <label>Experience Factor (0-100) <span className="required">*</span></label>
                      <input type="number" step="0.01" name="experience_factor" value={formData.experience_factor} onChange={handleChange} min="0" max="100" className={validationErrors.experience_factor ? 'error' : ''} />
                      {validationErrors.experience_factor && <span className="error-message">{validationErrors.experience_factor}</span>}
                    </div>
                    <div className="form-group">
                      <label>Recent Form (0-100) <span className="required">*</span></label>
                      <input type="number" step="0.01" name="recent_form_rating" value={formData.recent_form_rating} onChange={handleChange} min="0" max="100" className={validationErrors.recent_form_rating ? 'error' : ''} />
                      {validationErrors.recent_form_rating && <span className="error-message">{validationErrors.recent_form_rating}</span>}
                    </div>
                    <div className="form-group">
                      <label>Match-Winning Performances <span className="required">*</span></label>
                      <input type="number" name="match_winning_performances" value={formData.match_winning_performances} onChange={handleChange} min="0" className={validationErrors.match_winning_performances ? 'error' : ''} />
                      {validationErrors.match_winning_performances && <span className="error-message">{validationErrors.match_winning_performances}</span>}
                    </div>
                    <div className="form-group">
                      <label>Pressure Handling (0-100) <span className="required">*</span></label>
                      <input type="number" step="0.01" name="pressure_handling_score" value={formData.pressure_handling_score} onChange={handleChange} min="0" max="100" className={validationErrors.pressure_handling_score ? 'error' : ''} />
                      {validationErrors.pressure_handling_score && <span className="error-message">{validationErrors.pressure_handling_score}</span>}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <motion.button
              type="submit"
              className="predict-btn"
              disabled={loading}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              {loading ? (
                <span className="loading">
                  <span className="spinner"></span> Predicting...
                </span>
              ) : (
                'PREDICT AUCTION PRICE'
              )}
            </motion.button>
          </form>
        </motion.div>

        <AnimatePresence>
          {prediction && (
            <motion.div
              className="prediction-container"
              initial={{ opacity: 0, scale: 0.8, y: 50 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.8, y: 50 }}
              transition={{ duration: 0.5, type: "spring" }}
            >
              <div className="prediction-card">
                <motion.div
                  className="trophy-icon"
                  initial={{ rotate: -180, scale: 0 }}
                  animate={{ rotate: 0, scale: 1 }}
                  transition={{ delay: 0.2, duration: 0.6, type: "spring" }}
                >
                  🏆
                </motion.div>
                <h2 className="prediction-title">Predicted Auction Price</h2>
                
                {(() => {
                  const priceInCrores = convertToCrores(prediction.predicted_price);
                  const tierInfo = getPlayerTier(priceInCrores);
                  
                  return (
                    <>
                      <motion.div
                        className="tier-badge"
                        style={{
                          background: tierInfo.bgColor,
                          borderColor: tierInfo.color
                        }}
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                      >
                        <span className="tier-name" style={{ color: tierInfo.color }}>
                          {tierInfo.tier}
                        </span>
                        <span className="tier-description">{tierInfo.description}</span>
                      </motion.div>

                      <motion.div
                        className="price-display"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: 0.4, type: "spring", stiffness: 200 }}
                      >
                        <span className="currency">₹</span>
                        <span className="amount">{priceInCrores}</span>
                        <span className="unit">CRORE</span>
                      </motion.div>
                    </>
                  );
                })()}

                <div className="prediction-details">
                  <div className="detail-item">
                    <span className="detail-label">Confidence</span>
                    <div className="confidence-bar">
                      <motion.div
                        className="confidence-fill"
                        initial={{ width: 0 }}
                        animate={{ width: `${prediction.confidence}%` }}
                        transition={{ delay: 0.6, duration: 0.8 }}
                      />
                      <span className="confidence-text">{prediction.confidence}%</span>
                    </div>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Price Range</span>
                    <span className="detail-value">
                      ₹{convertToCrores(prediction.price_range.min)} - ₹{convertToCrores(prediction.price_range.max)} Cr
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <motion.footer
        className="footer"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
      >
        <p>Trained on 50,000+ synthetic player records | Naive Bayes Classification</p>
      </motion.footer>
    </div>
  );
}

export default App;

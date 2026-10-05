import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
const API_URL = process.env.REACT_APP_API_URL || 'https://ai-visa-eligibility-screening.onrender.com';

function App() {
  const [formData, setFormData] = useState({
    countryOfCitizenship: '',
    destinationCountry: '',
    purposeOfVisit: '',
    lengthOfStay: '',
    age: ''
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [provider, setProvider] = useState(null);

  // Check backend health on component mount
  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await axios.get(`${API_URL}/health`, { timeout: 60000 });
      if (response.data.status === 'healthy') {
        setBackendStatus('online');
      }
    } catch (err) {
      setBackendStatus('offline');
      console.error('Backend health check failed:', err);
    }
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    // Clear previous results when form changes
    if (result) {
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    setProvider(null);

    try {
      const response = await axios.post(
        `${API_URL}/check-eligibility`,
        formData,
        { timeout: 120000 } // 30 second timeout
      );
      setResult(response.data.eligibility);
      setProvider(response.data.provider || 'unknown');
    } catch (err) {
      if (err.code === 'ECONNABORTED') {
        setError('Request timed out. The server took too long to respond. Please try again.');
      } else if (err.response) {
        setError(err.response.data?.detail || 'Server error occurred. Please try again.');
      } else if (err.request) {
        setError('Unable to connect to the server. Please wait a moment and try again. The backend may be waking up.');
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData({
      countryOfCitizenship: '',
      destinationCountry: '',
      purposeOfVisit: '',
      lengthOfStay: '',
      age: ''
    });
    setResult(null);
    setError(null);
    setProvider(null);
  };

  return (
    <div className="App">
      <div className="header">
        <h1>🌍 Global Visa Navigator</h1>
        <p>Check Your Visa Eligibility Instantly</p>
        <div className="backend-status">
          {backendStatus === 'checking' && <span className="status-badge checking">🔄 Checking backend...</span>}
          {backendStatus === 'online' && <span className="status-badge online">✅ Backend Online</span>}
          {backendStatus === 'offline' && <span className="status-badge offline">⚠️ Backend Offline - Please start the server</span>}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="visa-form">
        <div className="form-group">
          <label htmlFor="countryOfCitizenship">Country of Citizenship</label>
          <input
            type="text"
            id="countryOfCitizenship"
            name="countryOfCitizenship"
            value={formData.countryOfCitizenship}
            onChange={handleChange}
            placeholder="e.g., India, USA, UK"
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="destinationCountry">Destination Country</label>
          <input
            type="text"
            id="destinationCountry"
            name="destinationCountry"
            value={formData.destinationCountry}
            onChange={handleChange}
            placeholder="e.g., Canada, Germany, UK"
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="purposeOfVisit">Purpose of Visit</label>
          <select
            id="purposeOfVisit"
            name="purposeOfVisit"
            value={formData.purposeOfVisit}
            onChange={handleChange}
            required
          >
            <option value="">Select purpose</option>
            <option value="Tourism">Tourism</option>
            <option value="Business">Business</option>
            <option value="Study">Study</option>
            <option value="Work">Work</option>
            <option value="Family Reunion">Family Reunion</option>
            <option value="Medical Treatment">Medical Treatment</option>
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="lengthOfStay">Length of Stay (days)</label>
          <input
            type="number"
            id="lengthOfStay"
            name="lengthOfStay"
            value={formData.lengthOfStay}
            onChange={handleChange}
            placeholder="e.g., 30"
            min="1"
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="age">Age</label>
          <input
            type="number"
            id="age"
            name="age"
            value={formData.age}
            onChange={handleChange}
            placeholder="e.g., 25"
            min="1"
            max="120"
            required
          />
        </div>

        <div className="button-group">
          <button type="submit" className="submit-btn" disabled={loading || backendStatus === 'offline'}>
            {loading ? '🔍 Analyzing...' : '✨ Check Eligibility'}
          </button>
          <button type="button" className="reset-btn" onClick={handleReset} disabled={loading}>
            🔄 Reset Form
          </button>
        </div>
      </form>

      {loading && (
        <div className="loading">
          <div className="spinner"></div>
          <p>🔍 Analyzing your visa eligibility...</p>
          <p className="loading-subtext">This may take a few seconds</p>
        </div>
      )}

      {error && (
        <div className="result-section error">
          <h2>❌ Error</h2>
          <p>{error}</p>
          <button className="retry-btn" onClick={() => window.location.reload()}>
            🔄 Retry
          </button>
        </div>
      )}

      {result && !error && (
        <div className="result-section success">
          <div className="result-header">
            <h2>✅ Eligibility Assessment Complete</h2>
            {provider && (
              <span className="provider-badge">
                {provider === 'openai' && '🤖 Powered by OpenAI GPT'}
                {provider === 'gemini' && '🌟 Powered by Google Gemini'}
                {provider === 'retrieval-only' && '📚 Document-Based Analysis'}
              </span>
            )}
          </div>
          <div className="result-content">
            {result}
          </div>
          <div className="result-actions">
            <button className="new-search-btn" onClick={handleReset}>
              🔍 New Search
            </button>
            <button className="copy-btn" onClick={() => navigator.clipboard.writeText(result)}>
              📋 Copy Result
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

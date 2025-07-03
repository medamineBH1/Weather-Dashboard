# 🌤️ Real-Time Weather Dashboard with AI Forecasting

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-dashboard-url.streamlit.app)

An advanced weather monitoring system that collects real-time global weather data, trains AI forecasting models, and provides interactive visualizations.

## Features
- Real-time weather data collection from OpenWeatherMap API
- Automatic hourly data updates
- Prophet-based AI forecasting models
- Interactive dashboard with maps and charts
- Email alert system for extreme weather conditions
- Historical data tracking

## 🛠️ Technical Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **ML Framework**: Prophet
- **Visualization**: Plotly, Folium
- **Data Storage**: Parquet files

## Installation
```bash
git clone https://github.com/yourusername/weather-dashboard.git
cd weather-dashboard
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Weather-Dashboard
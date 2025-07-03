import streamlit as st
import pandas as pd
import glob
import os
import plotly.express as px
import folium
from streamlit_folium import st_folium
from branca.colormap import linear
from datetime import datetime
import joblib
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly

# Set page configuration
st.set_page_config(
    page_title="Weather Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Weather icons mapping
WEATHER_ICONS = {
    "Clear": "☀️", "Clouds": "☁️", "Rain": "🌧️", "Drizzle": "🌦️", "Thunderstorm": "⛈️",
    "Snow": "❄️", "Mist": "🌫️", "Smoke": "💨", "Haze": "😶‍🌫️", "Dust": "💨", 
    "Fog": "🌁", "Sand": "🌪️", "Ash": "🌋", "Squall": "💨", "Tornado": "🌪️"
}

def load_latest_data():
    """Load the most recent weather data file"""
    try:
        # Find all Parquet files in weather_data directory
        parquet_files = glob.glob('weather_data/**/*.parquet', recursive=True)
        
        if not parquet_files:
            st.warning("No data files found. Please run your pipeline first!")
            return pd.DataFrame()
        
        # Get the latest file
        latest_file = max(parquet_files, key=os.path.getctime)
        df = pd.read_parquet(latest_file)
        
        # Add timestamp for display
        file_time = datetime.fromtimestamp(os.path.getctime(latest_file))
        df['last_updated'] = file_time.strftime("%Y-%m-%d %H:%M")
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def create_weather_map(df):
    """Create interactive weather map visualization"""
    if df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
        return None
    
    # Create base map centered on first city
    first_city = df.iloc[0]
    m = folium.Map(
        location=[first_city['latitude'], first_city['longitude']],
        zoom_start=2,
        tiles='CartoDB positron',
        control_scale=True
    )
    
    # Create temperature color scale
    min_temp = df['temperature'].min()
    max_temp = df['temperature'].max()
    colormap = linear.YlOrRd_09.scale(min_temp, max_temp)
    colormap.caption = 'Temperature (°C)'
    colormap.add_to(m)
    
    # Add markers for each city
    for _, row in df.iterrows():
        # Determine marker color based on temperature
        if row['temperature'] < 10:
            color = 'blue'
        elif row['temperature'] < 20:
            color = 'green'
        elif row['temperature'] < 30:
            color = 'orange'
        else:
            color = 'red'
        
        # Determine icon based on weather condition
        if 'Clear' in row['weather_condition']:
            icon = 'sun'
        elif 'Cloud' in row['weather_condition']:
            icon = 'cloud'
        elif 'Rain' in row['weather_condition']:
            icon = 'umbrella'
        elif 'Snow' in row['weather_condition']:
            icon = 'snowflake'
        else:
            icon = 'info-sign'
        
        # Create popup content
        popup_html = f"""
        <div style="width: 250px; font-family: Arial, sans-serif;">
            <h3 style="margin: 5px 0; color: #333;">{row['city']}</h3>
            <div style="display: flex; align-items: center; margin: 10px 0;">
                <span style="font-size: 24px; margin-right: 10px;">{WEATHER_ICONS.get(row['weather_condition'], '❓')}</span>
                <span style="font-size: 20px; font-weight: bold;">{row['temperature']}°C</span>
            </div>
            <p style="margin: 5px 0;"><b>Condition:</b> {row['weather_condition']}</p>
            <p style="margin: 5px 0;"><b>Humidity:</b> {row['humidity']}%</p>
            <p style="margin: 5px 0;"><b>Wind:</b> {row['wind_speed']} km/h</p>
            <p style="margin: 5px 0;"><b>Pressure:</b> {row['pressure']} hPa</p>
            <p style="margin: 5px 0; font-size: 12px; color: #666;">Updated: {row.get('last_updated', 'N/A')}</p>
        </div>
        """
        
        # Add marker to map
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['city']}: {row['temperature']}°C",
            icon=folium.Icon(
                icon=icon,
                color='white',
                icon_color=color,
                prefix='fa'
            )
        ).add_to(m)
    
    return m

def load_forecast_model():
    """Load the latest forecast model"""
    try:
        with open("models/latest_model.txt") as f:
            model_path = f.read().strip()
        return joblib.load(model_path)
    except:
        return None

def generate_forecast(model, periods=24):
    """Generate future forecast"""
    future = model.make_future_dataframe(periods=periods, freq='H')
    forecast = model.predict(future)
    return forecast

# Create dashboard
st.title('🌤️ Advanced Weather Dashboard')
df = load_latest_data()

# Load forecast models
forecast_models = load_forecast_model()

if not df.empty:
    # Add weather icons
    df['icon'] = df['weather_condition'].map(WEATHER_ICONS).fillna('❓')
    
    # ===== SIDEBAR CONTROLS =====
    st.sidebar.title("Dashboard Controls")
    
    # City selection
    selected_cities = st.sidebar.multiselect(
        "Select Cities",
        options=df['city'].unique(),
        default=df['city'].unique()
    )
    
    # Temperature unit toggle
    unit = st.sidebar.radio(
        "Temperature Unit",
        ["°C", "°F"],
        index=0
    )
    
    # Convert temperature if needed
    if unit == "°F":
        df['temperature'] = (df['temperature'] * 9/5) + 32
    
    # Filter data based on selections
    filtered_df = df[df['city'].isin(selected_cities)]
    
    # Last updated info
    last_update = filtered_df['last_updated'].iloc[0] if 'last_updated' in filtered_df.columns else "N/A"
    st.sidebar.markdown(f"**Last Updated:** {last_update}")
    
    # ===== MAIN DASHBOARD =====
    
    # Current conditions cards
    st.subheader('Current Conditions')
    
    # Create columns dynamically based on number of cities
    cols = st.columns(len(filtered_df))
    
    for i, (_, row) in enumerate(filtered_df.iterrows()):
        with cols[i]:
            # City header with icon
            st.markdown(f"### {row['icon']} {row['city']}")
            
            # Temperature metric
            st.metric(
                label="Temperature",
                value=f"{row['temperature']:.1f}{unit}",
                help=f"Feels like {row['temperature']:.1f}{unit}"
            )
            
            # Additional metrics
            st.progress(row['humidity']/100, text=f"💧 Humidity: {row['humidity']}%")
            st.write(f"💨 Wind: {row['wind_speed']} km/h")
            st.write(f"📊 Pressure: {row['pressure']} hPa")
    
    # Weather map visualization
    st.markdown("---")
    st.subheader('Interactive Weather Map')
    if 'latitude' in df.columns and 'longitude' in df.columns:
        weather_map = create_weather_map(filtered_df)
        if weather_map:
            st_folium(weather_map, width=1200, height=500)
        else:
            st.warning("Map data not available for selected cities")
    else:
        st.warning("Geographic coordinates not found in data. Please update your pipeline to include latitude/longitude.")
    
    # Charts section
    st.markdown("---")
    
    # Temperature comparison chart
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader('Temperature Comparison')
        fig_temp = px.bar(
            filtered_df.sort_values('temperature'),
            x='city',
            y='temperature',
            color='temperature',
            color_continuous_scale='thermal',
            text_auto=True,
            height=400
        )
        fig_temp.update_layout(
            yaxis_title=f"Temperature ({unit})",
            xaxis_title="City"
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        st.subheader('Weather Distribution')
        weather_counts = filtered_df['weather_condition'].value_counts().reset_index()
        weather_counts.columns = ['Condition', 'Count']
        
        fig_pie = px.pie(
            weather_counts,
            names='Condition',
            values='Count',
            hole=0.3,
            height=300
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Humidity and wind charts
    st.subheader('Detailed Metrics')
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**Humidity Levels**")
        fig_humidity = px.bar(
            filtered_df.sort_values('humidity'),
            x='city',
            y='humidity',
            color='humidity',
            color_continuous_scale='blues',
            height=300
        )
        st.plotly_chart(fig_humidity, use_container_width=True)
    
    with col4:
        st.markdown("**Wind Speed**")
        fig_wind = px.bar(
            filtered_df.sort_values('wind_speed'),
            x='city',
            y='wind_speed',
            color='wind_speed',
            color_continuous_scale='greens',
            height=300
        )
        st.plotly_chart(fig_wind, use_container_width=True)
    
    # Raw data table
    st.markdown("---")
    st.subheader('Raw Data')
    st.dataframe(
        filtered_df.drop(columns=['icon'], errors='ignore')
        .style.background_gradient(subset=['temperature', 'humidity', 'wind_speed'], cmap='coolwarm'),
        height=300
    )
    
    # ===== AI FORECAST SECTION =====
    st.markdown("---")
    st.subheader("AI-Powered Weather Forecast")
    
    if forecast_models:
        forecast_city = st.selectbox(
            "Select city for forecast", 
            options=df['city'].unique(),
            index=0
        )
        
        if forecast_city in forecast_models:
            model = forecast_models[forecast_city]
            
            # Generate forecast
            forecast = generate_forecast(model)
            
            # Show forecast chart
            fig_forecast = plot_plotly(model, forecast)
            fig_forecast.update_layout(
                title=f"24-Hour Temperature Forecast for {forecast_city}",
                xaxis_title="Date/Time",
                yaxis_title=f"Temperature ({unit})"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Show forecast components
            st.subheader("Forecast Components")
            components_fig = model.plot_components(forecast)
            st.pyplot(components_fig)
        else:
            st.warning(f"No forecast model available for {forecast_city}")
    else:
        st.warning("No forecast models available. Train models using train_models.py")
        
else:
    st.warning("No weather data available. Please run your pipeline first.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Weather data from OpenWeatherMap API")
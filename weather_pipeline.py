import requests
import pandas as pd
import os
import time
import schedule
import smtplib
import csv
from datetime import datetime
from dotenv import load_dotenv
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Load environment variables
load_dotenv()

# API Configuration
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
CITIES = ["London", "New York", "Tokyo", "Sydney", "Cairo", "Tunis", "Sousse", "Medenine", "Moscow", "Beijing"]
UNITS = "metric"
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Email Configuration
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
EMAIL_SENDER = os.getenv("ALERT_EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD")
EMAIL_RECIPIENT = os.getenv("ALERT_EMAIL_RECIPIENT")

def parse_float(value, default):
    """Safely parse float values from environment"""
    try:
        return float(value.strip().split()[0])  
    except (ValueError, TypeError):
        return default

HEAT_THRESHOLD = parse_float(os.getenv("HEAT_WAVE_THRESHOLD"), 35)
COLD_THRESHOLD = parse_float(os.getenv("COLD_WAVE_THRESHOLD"), 0)

def reload_env():
    """Reload environment variables from .env file."""
    load_dotenv(override=True)
    return {
        "SMTP_SERVER": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
        "SMTP_PORT": int(os.getenv("SMTP_PORT", 587)),
        "EMAIL_SENDER": os.getenv("ALERT_EMAIL_SENDER"),
        "EMAIL_PASSWORD": os.getenv("ALERT_EMAIL_PASSWORD"),
        "EMAIL_RECIPIENT": os.getenv("ALERT_EMAIL_RECIPIENT"),
        "HEAT_THRESHOLD": parse_float(os.getenv("HEAT_WAVE_THRESHOLD"), 35),
        "COLD_THRESHOLD": parse_float(os.getenv("COLD_WAVE_THRESHOLD"), 0)
    }

def fetch_weather_data():
    """Fetch current weather for all cities"""
    records = []
    for city in CITIES:
        params = {"q": city, "appid": API_KEY, "units": UNITS}
        try:
            print(f"Fetching {city}...", end="", flush=True)
            response = requests.get(BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            records.append({
                "city": city,
                "timestamp": datetime.utcfromtimestamp(data["dt"]).strftime('%Y-%m-%d %H:%M:%S'),
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"],
                "weather_condition": data["weather"][0]["main"],
                "pressure": data["main"]["pressure"],
                "feels_like": data["main"]["feels_like"],
                "latitude": data["coord"]["lat"],
                "longitude": data["coord"]["lon"]
            })
            print(" done")
            time.sleep(1)
        except Exception as e:
            print(f" error: {str(e)}")
    return pd.DataFrame(records)

def save_training_data(df):
    """Append data to training CSV for model training"""
    try:
        training_path = "weather_data/training_dataset.csv"
        file_exists = os.path.exists(training_path)
        
        with open(training_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['city', 'timestamp', 'temperature'])
            for _, row in df.iterrows():
                writer.writerow([row['city'], row['timestamp'], row['temperature']])
        return True
    except Exception as e:
        print(f"❌ Failed to save training data: {str(e)}")
        return False

def save_data(df):
    """Save data with timestamp and append to historical dataset"""
    os.makedirs("weather_data", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Save current snapshot
    current_path = f"weather_data/weather_{timestamp}.parquet"
    df.to_parquet(current_path, engine='pyarrow')
    
    # Update historical data
    historical_path = "weather_data/historical.parquet"
    if os.path.exists(historical_path):
        historical_df = pd.read_parquet(historical_path)
        updated_df = pd.concat([historical_df, df])
        updated_df.to_parquet(historical_path)
    else:
        df.to_parquet(historical_path)
    
    return current_path

def send_email_alert(settings, subject, body):
    """Send email alert with weather information"""
    if not settings["EMAIL_SENDER"] or not settings["EMAIL_PASSWORD"] or not settings["EMAIL_RECIPIENT"]:
        print("⚠️ Email alerts not configured. Skipping.")
        return
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = settings["EMAIL_SENDER"]
        msg['To'] = settings["EMAIL_RECIPIENT"]
        msg['Subject'] = subject
        
        # HTML email content
        html = f"""<html>
            <body>
                <h2>Weather Alert!</h2>
                <p>{body}</p>
                <p><strong>Next check in 1 hour</strong></p>
                <p style="color: #888; font-size: 12px;">
                    Sent from Weather Monitoring System
                </p>
            </body>
        </html>"""
        
        msg.attach(MIMEText(html, "html"))
        
        # Send email
        with smtplib.SMTP(settings["SMTP_SERVER"], settings["SMTP_PORT"]) as server:
            server.starttls()
            server.login(settings["EMAIL_SENDER"], settings["EMAIL_PASSWORD"])
            server.send_message(msg)
            print(f"✅ Email alert sent to {settings['EMAIL_RECIPIENT']}")
    except Exception as e:
        print(f"❌ Failed to send email: {str(e)}")

def check_alerts(weather_df):
    """Check for extreme weather conditions"""
    if os.getenv("ENABLE_ALERTS", "false").lower() != "true":
        print("⚠️ Alerts are currently disabled in configuration")
        return False
    
    settings = reload_env()
    alerts = []
    
    # Heat wave alerts
    heat_wave = weather_df[weather_df['temperature'] > settings["HEAT_THRESHOLD"]]
    if not heat_wave.empty:
        cities = ", ".join(heat_wave['city'].tolist())
        alerts.append({
            "type": "heat",
            "message": f"Heat wave alert! {cities} above {settings['HEAT_THRESHOLD']}°C"
        })
    
    # Cold wave alerts
    cold_wave = weather_df[weather_df['temperature'] < settings["COLD_THRESHOLD"]]
    if not cold_wave.empty:
        cities = ", ".join(cold_wave['city'].tolist())
        alerts.append({
            "type": "cold",
            "message": f"Cold wave alert! {cities} below {settings['COLD_THRESHOLD']}°C"
        })
    
    # Send alerts
    for alert in alerts:
        print(f"⚠️ ALERT: {alert['message']}")
        send_email_alert(
            settings,
            subject=f"Weather Alert: {alert['type'].title()} Wave",
            body=alert['message']
        )
    
    return len(alerts) > 0

def run_pipeline():
    """Execute full pipeline run"""
    start_time = datetime.now()
    print(f"\n=== Pipeline started at {start_time.strftime('%Y-%m-%d %H:%M')} ===")
    
    try:
        weather_df = fetch_weather_data()
        if not weather_df.empty:
            # Save to persistent storage
            save_path = save_data(weather_df)
            print(f"✅ Data saved to {save_path}")
            
            # Save to training dataset
            if save_training_data(weather_df):
                print("✅ Training data appended")
            
            # Check for alerts
            if os.getenv("ENABLE_ALERTS", "false").lower() == "true":
                check_alerts(weather_df)
            
            duration = (datetime.now() - start_time).seconds
            print(f"=== Pipeline completed in {duration}s ===")
            return True
        print("⚠️ No data retrieved")
        return False
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        return False

def main():
    """Main execution function"""
    # Initial run
    run_pipeline()
    
    # Schedule hourly runs
    schedule.every().hour.do(run_pipeline)
    
    print("Scheduler started. Runs will execute hourly. Press Ctrl+C to exit.")
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
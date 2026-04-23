import pymongo
import random
import time
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = MONGO_URI = os.getenv("MONGO_URI") 
client = pymongo.MongoClient(MONGO_URI)
db = client["supply_chain"]
events_collection = db["supply_chain_data"]

TARGET_SKU = "22197"
LOCATIONS = [
    "Northampton Fulfillment Centre", 
    "Warrington Regional Cross-Dock",  
    "Dartford Last-Mile Depot",       
    "Felixstowe Port Intake",          
    "M1 Transit Corridor"
]

# --- 2. LEARN THE TRUE DISTRIBUTION (KDE) ---
print("Analyzing historical data distribution...")
try:
    df = pd.read_csv('supply_chain_3yr_data.csv')
    historical_sales = df[TARGET_SKU].dropna().values
    
    # Filter out extreme outliers and absolute zeros just to get the core "normal" shape
    historical_sales = historical_sales[(historical_sales > 0) & (historical_sales < np.percentile(historical_sales, 99))]
    
    # Fit the Kernel Density Estimate (This maps the exact shape of your real data)
    real_distribution_kde = stats.gaussian_kde(historical_sales)
    
    # Calculate bounds for our anomalies
    p95_val = np.percentile(historical_sales, 95)
    p99_val = np.percentile(historical_sales, 99)
    
    print("Distribution mapped successfully. Ready to stream.")
except Exception as e:
    print(f"Error loading historical data: {e}")
    print("Ensure 'supply_chain_3yr_data.csv' is in the same directory.")
    exit()

# --- 3. EVENT GENERATOR ---
def generate_event(timestamp=None):
    """Generates an event mathematically aligned with the real data distribution."""
    if timestamp is None:
        timestamp = datetime.now()
        
    event_roll = random.random()

    qty = 0 
    
    if event_roll < 0.85:
        event_type = "normal_sale"
        severity = 1
        qty = int(np.maximum(0, real_distribution_kde.resample(1)[0][0]))
        desc = f"Routine daily sale of {qty} units."
        
    elif event_roll < 0.90:
        event_type = "restock"
        severity = 1
        qty = int(p99_val * random.uniform(3.0, 5.0))
        desc = f"Successful inbound shipment of {qty} units."
        
    elif event_roll < 0.95:
        event_type = "delay"
        severity = random.randint(2, 3)
        desc = "Weather delay on inbound transit route."
        
    elif event_roll < 0.98:
        event_type = "anomaly"
        severity = random.randint(3, 4)
        qty = int(p99_val * random.uniform(1.2, 2.0))
        desc = f"Demand anomaly detected! {qty} units requested, exceeding P99 historical variance."
        
    else:
        event_type = "stockout"
        severity = 5
        desc = "CRITICAL: Inventory dropped to zero. Missed sales occurring."

    return {
            "timestamp": timestamp,
            "product_id": TARGET_SKU,
            "event_type": event_type,
            "severity": severity,
            "location": random.choice(LOCATIONS),
            "description": desc,
            "quantity": qty
        }


# --- 4. EXECUTION ENGINES ---
def seed_historical_data(days_back=14):
    """Fills MongoDB with past data so the chatbot has history to query."""
    print(f"Adding historical data to database with {days_back} days of historical events...")
    events_collection.delete_many({"product_id": TARGET_SKU}) 
    
    historical_events = []
    start_time = datetime.now() - timedelta(days=days_back)
    
    for i in range(days_back * 10):
        simulated_time = start_time + timedelta(hours=(i * 2.4) + random.uniform(-1, 1))
        historical_events.append(generate_event(simulated_time))
        
    events_collection.insert_many(historical_events)
    print(f"Successfully inserted {len(historical_events)} events.")

def run_live_stream(interval_seconds=5):
    """Simulates a live data stream."""    
    try:
        while True:
            new_event = generate_event()
            events_collection.insert_one(new_event)
            
            if new_event["severity"] >= 3:
                print(f"ALERT [{new_event['timestamp'].strftime('%H:%M:%S')}]: {new_event['event_type'].upper()} - {new_event['description']}")
            else:
                print(f"LOG [{new_event['timestamp'].strftime('%H:%M:%S')}]: {new_event['event_type']} processed.")
                
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        print("\nLive stream stopped by user.")

if __name__ == "__main__":
    seed_historical_data(days_back=14)
    run_live_stream(interval_seconds=10)




st.markdown("### 📊 Live Distribution Monitor (KS-Test Visualizer)")

@st.cache_data
def load_historical_base():
    """Loads the real 3-year history so we have a baseline to compare against."""
    df = pd.read_csv('supply_chain_3yr_data.csv')
    hist_sales = df['22197'].dropna().values 
    return hist_sales[(hist_sales > 0) & (hist_sales < np.percentile(hist_sales, 99))]

hist_sales = load_historical_base()

mongo_client = pymongo.MongoClient(MONGO_URI)
db = mongo_client["supply_chain_db"]
events_collection = db["emergency_events"]

plot_placeholder = st.empty()

live_sync = st.checkbox("🔄 Enable Live Database Sync (Updates every 5s)")

if live_sync:
    cursor = events_collection.find({"quantity": {"$gt": 0}})
    synthetic_data = [doc['quantity'] for doc in cursor]

    if len(synthetic_data) > 0:
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=hist_sales, 
            name='Historical Real Data', 
            histnorm='probability density', 
            opacity=0.5, 
            marker_color='gray'
        ))
        
        fig.add_trace(go.Histogram(
            x=synthetic_data, 
            name='Live Synthetic Stream', 
            histnorm='probability density', 
            opacity=0.7, 
            marker_color='#1f77b4'
        ))

        fig.update_layout(
            barmode='overlay', 
            title='Real vs. Synthetic Data Distribution (Density Overlay)', 
            xaxis_title='Units Sold per Event', 
            yaxis_title='Probability Density',
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        
        plot_placeholder.plotly_chart(fig, use_container_width=True)

    time.sleep(5)
    st.rerun()
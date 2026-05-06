import json
import random
import time
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

TARGET_SKU = "22197"
LOCATIONS = [
    "Northampton Fulfillment Centre", 
    "Warrington Regional Cross-Dock",  
    "Dartford Last-Mile Depot",       
    "Felixstowe Port Intake",          
    "M1 Transit Corridor"
]

print("Analyzing historical data distribution...")
try:
    df = pd.read_csv('supply_chain_3yr_data.csv')
    historical_sales = df[TARGET_SKU].dropna().values
    historical_sales = historical_sales[(historical_sales > 0) & (historical_sales < np.percentile(historical_sales, 99))]
    
    real_distribution_kde = stats.gaussian_kde(historical_sales)
    p99_val = np.percentile(historical_sales, 99)
    print("Distribution mapped successfully. Ready to stream.")
except Exception as e:
    print(f"Error: {e}")
    print("Make sure 'supply_chain_3yr_data.csv' is uploaded to this EC2 instance!")
    exit()

def generate_event():
    timestamp_str = datetime.now().isoformat()
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
            "timestamp": timestamp_str, # Passed as string for Kafka
            "product_id": TARGET_SKU,
            "event_type": event_type,
            "severity": severity,
            "location": random.choice(LOCATIONS),
            "description": desc,
            "quantity": qty
        }

print("\nStarting LIVE Agentic Kafka Stream...")
try:
    while True:
        new_event = generate_event()
        producer.send('uk-logistics-stream', value=new_event)
        
        if new_event["severity"] >= 3:
            print(f"PUBLISHED: {new_event['event_type'].upper()} - {new_event['description']}")
        else:
            print(f"PUBLISHED: {new_event['event_type']} processed.")
            
        time.sleep(5) # Adjust stream speed here
        
except KeyboardInterrupt:
    print("\nLive stream stopped by user.")
    producer.close()
import pandas as pd
import random
from datetime import datetime
import pymongo
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
TARGET_SKU = "22197"
CSV_FILE = "model_v7_2026_forecast.csv"
CURRENT_DATE = pd.to_datetime("2026-05-06")

LOCATIONS = [
    "Northampton Fulfillment Centre", 
    "Warrington Regional Cross-Dock",  
    "Dartford Last-Mile Depot",       
    "Felixstowe Port Intake",          
    "M1 Transit Corridor"
]

def sync_mongodb_to_baseline():
    print("🔗 Connecting to MongoDB...")
    client = pymongo.MongoClient(MONGO_URI)
    db = client["supply_chain"]
    collection = db["supply_chain_data"]

    print("🧹 Clearing old chaotic data...")
    collection.delete_many({})

    print(f"📊 Loading golden baseline from {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
    except FileNotFoundError:
        print(f"❌ Error: '{CSV_FILE}' not found. Ensure it is in the same directory.")
        return

    df_past = df[df['Date'] <= CURRENT_DATE]
    mongo_docs = []

    print(f"⚙️ Generating synchronized daily sales up to {CURRENT_DATE.strftime('%Y-%m-%d')}...")
    
    for _, row in df_past.iterrows():
        qty = int(row['Actual_Sales'])
        
        mongo_docs.append({
            "timestamp": row['Date'].to_pydatetime(),
            "event_type": "normal_sale",
            "product_id": TARGET_SKU,
            "severity": 1,
            "location": random.choice(LOCATIONS),
            "description": "Routine daily transaction volume",
            "quantity": qty
        })

    
    base_qty = mongo_docs[-1]["quantity"]
    
    #anomalies for the demo
    anomalies = [
        {
            "date": datetime(2026, 4, 28), 
            "qty": int(base_qty * 1.8), 
            "type": "anomaly", 
            "desc": "Viral social media trend caused sudden demand spike exceeding P95 risk bounds.", 
            "sev": 4
        },
        {
            "date": datetime(2026, 5, 2), 
            "qty": int(base_qty * 2.1), 
            "type": "anomaly", 
            "desc": "Unanticipated B2B bulk order placed bypassing standard wholesale channels.", 
            "sev": 4
        },
        {
            "date": datetime(2026, 5, 4), 
            "qty": 0, 
            "type": "stockout", 
            "desc": "CRITICAL: Inventory dropped to zero. Supplier truck delayed on M1 Transit Corridor.", 
            "sev": 5
        }
    ]

    for anom in anomalies:
        mongo_docs.append({
            "timestamp": anom["date"],
            "event_type": anom["type"],
            "product_id": TARGET_SKU,
            "severity": anom["sev"],
            "location": random.choice(LOCATIONS),
            "description": anom["desc"],
            "quantity": anom["qty"]
        })

    print(f"Pushing {len(mongo_docs)} records to MongoDB...")
    collection.insert_many(mongo_docs)
    print("Success! MongoDB is now perfectly synchronized with your 2026 forecast.")

if __name__ == "__main__":
    sync_mongodb_to_baseline()
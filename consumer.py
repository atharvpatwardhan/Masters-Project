import json
import pymongo
import os
from datetime import datetime
from dotenv import load_dotenv
from kafka import KafkaConsumer

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI") 
client = pymongo.MongoClient(MONGO_URI)
db = client["supply_chain"]
events_collection = db["supply_chain_data"]

EC2_PUBLIC_IP = '54.160.176.117'

consumer = KafkaConsumer(
    'uk-logistics-stream',
    bootstrap_servers=[f'{EC2_PUBLIC_IP}:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest'
)

print(f"Connected to Kafka at {EC2_PUBLIC_IP}.")

for message in consumer:
    event = message.value
    
    event['timestamp'] = datetime.fromisoformat(event['timestamp'])
    
    events_collection.insert_one(event)
    
    if event["severity"] >= 3:
        print(f"DB WRITE [ANOMALY]: {event['location']} - {event['description']}")
    else:
        print(f"DB WRITE [NORMAL]: {event['location']} - {event['quantity']} units")
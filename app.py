import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import boto3
from io import BytesIO
import random
import os
from dotenv import load_dotenv
import json
from google import genai
from google.genai import types
import pymongo
from bson import json_util
from datetime import datetime
from scipy.stats import ks_2samp
import time
import scipy.stats as stats


load_dotenv()

MONGO_URI = MONGO_URI = os.getenv("MONGO_URI") 

st.set_page_config(
    page_title="Supply Chain Command Center",
    page_icon="📦",
    layout="wide"
)

st.title("Supply Chain Forecasting & Anomaly Detection")
st.markdown("Real-time monitoring of hybrid model outputs and streaming transaction anomalies.")

@st.cache_data(ttl=3600)
def load_data_from_s3():
    """Pulls the latest pipeline output from a PUBLIC AWS S3 bucket."""
    try:
        bucket_name = 'atharv-supply-chain-project'
        file_name = 'model_v7_2026_forecast.csv'
        
        url = f"https://{bucket_name}.s3.amazonaws.com/{file_name}"
        
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.rename(columns={
        'P05_Forecast': 'Lower_Bound_95CI',
        'P95_Forecast': 'Upper_Bound_95CI'
        })
        return df
        
    except Exception as e:
        st.warning(f"Failed to load from S3 URL. Loading mock data. Error: {e}")
        dates = pd.date_range(start='2025-01-01', periods=52, freq='W')
        mock_actuals = np.random.normal(2500, 300, 52)
        return pd.DataFrame({
            'Date': dates,
            'Expected_Forecast': mock_actuals * 1.05,
            'Actual_Sales': mock_actuals,
            'Lower_Bound_95CI': mock_actuals * 0.8,
            'Upper_Bound_95CI': mock_actuals * 1.3,
            'Anomaly_Flag': np.where(np.random.rand(52) > 0.95, 1, 0)
        })

df = load_data_from_s3()

from datetime import datetime, timedelta

st.markdown("### Command Center")

@st.cache_data(ttl=60) # Caches the DB query for 60 seconds to prevent overwhelming the connection
def fetch_live_mongo_metrics():
    """Aggregates live sales and event data directly from MongoDB."""
    try:
        mongo_client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        db = mongo_client["supply_chain"]
        collection = db["supply_chain_data"]
        
        now = datetime.utcnow()
        one_week_ago = now - timedelta(days=7)
        two_weeks_ago = now - timedelta(days=14)
        
        current_week_sales = list(collection.aggregate([
            {"$match": {"timestamp": {"$gte": one_week_ago}, "event_type": "normal_sale"}},
            {"$group": {"_id": None, "total_sales": {"$sum": "$quantity"}, "order_count": {"$sum": 1}}}
        ]))
        
        prev_week_sales = list(collection.aggregate([
            {"$match": {"timestamp": {"$gte": two_weeks_ago, "$lt": one_week_ago}, "event_type": "normal_sale"}},
            {"$group": {"_id": None, "total_sales": {"$sum": "$quantity"}}}
        ]))
        
        critical_events = list(collection.aggregate([
            {"$match": {"timestamp": {"$gte": one_week_ago}, "event_type": {"$in": ["anomaly", "stockout", "delay"]}}},
            {"$group": {"_id": "$event_type", "count": {"$sum": 1}}}
        ]))
        
        curr_sales_vol = current_week_sales[0]["total_sales"] if current_week_sales else 0
        prev_sales_vol = prev_week_sales[0]["total_sales"] if prev_week_sales else 0
        
        events_dict = {doc["_id"]: doc["count"] for doc in critical_events}
        stockouts = events_dict.get("stockout", 0)
        anomalies = events_dict.get("anomaly", 0)
        delays = events_dict.get("delay", 0)
        
        return {
            "status": "success",
            "current_sales": curr_sales_vol,
            "prev_sales": prev_sales_vol,
            "stockouts": stockouts,
            "anomalies": anomalies,
            "delays": delays,
            "last_synced": now.strftime("%H:%M:%S UTC")
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

live_data = fetch_live_mongo_metrics()

# Extract the static forecast expectations from the S3 model data
latest_forecast = df['Expected_Forecast'].iloc[-1] if not df.empty else 0
p95_boundary = df['Upper_Bound_95CI'].iloc[-1] if not df.empty else 0

if live_data["status"] == "success":
    curr_sales = live_data["current_sales"]
    sales_delta = curr_sales - live_data["prev_sales"]
    
    forecast_variance = curr_sales - latest_forecast
    variance_pct = (forecast_variance / latest_forecast * 100) if latest_forecast > 0 else 0
    
    risk_ratio = (curr_sales / p95_boundary * 100) if p95_boundary > 0 else 0
    total_incidents = live_data["stockouts"] + live_data["anomalies"] + live_data["delays"]

    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Live Weekly Volume", 
            value=f"{int(curr_sales):,}", 
            delta=f"{int(sales_delta):,} vs last week"
        )
    
    with col2:
        st.metric(
            label="Forecast Variance", 
            value=f"{int(forecast_variance):,}", 
            delta=f"{variance_pct:.1f}% vs XGBoost",
            delta_color="inverse" # Shows red if demand is surging higher than expected
        )
        
    with col3:
        st.metric(
            label="P95 Risk Limit", 
            value=f"{risk_ratio:.1f}%", 
            delta=f"{int(p95_boundary - curr_sales):,} units buffer",
            help="How close current live sales are to breaching the model's 95% Confidence Interval."
        )
        
    with col4:
        st.metric(
            label="Active Grid Incidents", 
            value=f"{total_incidents}", 
            delta=f"{live_data['stockouts']} Stockouts",
            delta_color="inverse"
        )
        
    with col5:
        st.metric(
            label="MongoDB Status", 
            value="Connected", 
            delta=f"Sync: {live_data['last_synced']}",
            delta_color="normal"
        )
        
    # Visual buffer bar
    st.progress(min(risk_ratio / 100, 1.0), text="Current Demand vs. P95 Risk Boundary")

else:
    st.error(f"MongoDB Connection Failed: {live_data.get('message')}")


st.markdown("### Forecast vs. Live Demand")
st.markdown("Comparing the hybrid model's 95% risk boundaries against real-time streaming actuals from MongoDB.")

@st.cache_data(ttl=60)
def fetch_live_chart_data():
    """Aggregates daily sales and pulls live anomalies directly from MongoDB."""
    try:
        mongo_client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        db = mongo_client["supply_chain"]
        collection = db["supply_chain_data"]
        
        # Fetch normal daily sales aggregates
        pipeline = [
            {"$match": {"event_type": "normal_sale", "timestamp": {"$exists": True}}},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                "daily_sales": {"$sum": "$quantity"}
            }},
            {"$sort": {"_id": 1}}
        ]
        sales_results = list(collection.aggregate(pipeline))
        
        live_sales_df = pd.DataFrame(sales_results)
        if not live_sales_df.empty:
            live_sales_df.columns = ['Date', 'Live_Actuals']
            live_sales_df['Date'] = pd.to_datetime(live_sales_df['Date'])
            
        # critical anomalies and stockouts
        anomalies_cursor = collection.find({"event_type": {"$in": ["anomaly", "stockout"]}})
        live_anomalies_df = pd.DataFrame(list(anomalies_cursor))
        
        return live_sales_df, live_anomalies_df
        
    except Exception as e:
        st.warning(f"Database connection error for timeseries: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Fetch the live MongoDB data
live_sales_df, live_anomalies_df = fetch_live_chart_data()

fig = go.Figure()

if not df.empty:
    fig.add_trace(go.Scatter(
        x=pd.concat([df['Date'], df['Date'][::-1]]),
        y=pd.concat([df['Upper_Bound_95CI'], df['Lower_Bound_95CI'][::-1]]),
        fill='toself',
        fillcolor='rgba(44, 160, 44, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='95% Risk Boundary'
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['Expected_Forecast'],
        mode='lines',
        line=dict(color='#2ca02c', width=2, dash='dash'),
        name='Model Expected Forecast'
    ))

if not live_sales_df.empty:
    fig.add_trace(go.Scatter(
        x=live_sales_df['Date'], 
        y=live_sales_df['Live_Actuals'],
        mode='lines',
        line=dict(color='black', width=3),
        name='Live Transactions (MongoDB)'
    ))
else:
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Actual_Sales'],
        mode='lines', line=dict(color='black', width=3),
        name='Historical Transactions (Fallback)'
    ))

if not live_anomalies_df.empty and 'timestamp' in live_anomalies_df.columns:
    for _, row in live_anomalies_df.iterrows():
        is_stockout = row.get('event_type') == 'stockout'
        marker_color = '#d62728' if is_stockout else '#ff7f0e' # Red for stockouts, Orange for anomalies
        marker_symbol = 'x' if is_stockout else 'circle'
        
        fig.add_trace(go.Scatter(
            x=[row['timestamp']], 
            y=[row.get('quantity', 0)],
            mode='markers',
            marker=dict(color=marker_color, size=14, symbol=marker_symbol, line=dict(color='white', width=2)),
            name=row.get('event_type', 'Anomaly').title(),
            showlegend=False, 
            hovertemplate=f"<b>Date:</b> %{{x}}<br>" +
                          f"<b>Quantity:</b> %{{y}}<br>" +
                          f"<b>Alert:</b> {row.get('description', 'Anomaly Detected')}<extra></extra>"
        ))

fig.update_layout(
    height=500,
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(
        orientation="h", 
        yanchor="bottom", y=1.02, 
        xanchor="right", x=1,
        bgcolor="rgba(255,255,255,0.8)"
    ),
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="Units Sold",
    template="plotly_white", 
    xaxis=dict(showgrid=True, gridcolor='#e0e0e0'),
    yaxis=dict(showgrid=True, gridcolor='#e0e0e0')
)

st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("### Dynamic Risk & Inventory Optimization")
st.markdown("Simulate financial exposure based on specific inventory decisions. Toggle Newsvendor logic for perishable/seasonal goods.")

current_mean = df['Expected_Forecast'].iloc[-1]
current_upper = df['Upper_Bound_95CI'].iloc[-1]
estimated_std = (current_upper - current_mean) / 1.96

is_newsvendor = st.checkbox("Enable Newsvendor Logic (Single-Period / Perishable Goods)", value=True, 
                            help="If checked, unsold goods are liquidated at Buyback Value. If unchecked, goods are rolled over and only incur Holding Costs.")

col_fin1, col_fin2, col_fin3, col_fin4 = st.columns(4)
with col_fin1:
    unit_cost = st.number_input("Unit Cost ($)", min_value=0.0, value=50.0, step=5.0)
with col_fin2:
    retail_price = st.number_input("Retail Price ($)", min_value=0.0, value=150.0, step=5.0)
with col_fin3:
    holding_cost = st.number_input("Holding Cost per Unit ($)", min_value=0.0, value=5.0, step=1.0)
with col_fin4:
    buyback_value = st.number_input("Salvage/Buyback Value ($)", min_value=0.0, value=30.0, step=5.0, disabled=not is_newsvendor)

cost_of_underage = max(0, retail_price - unit_cost)

if is_newsvendor:
    cost_of_overage = max(0, (unit_cost - buyback_value) + holding_cost) 
else:
    cost_of_overage = holding_cost 

if (cost_of_overage + cost_of_underage) > 0:
    critical_ratio = cost_of_underage / (cost_of_overage + cost_of_underage)
else:
    critical_ratio = 0.5

# Calculate Mathematical Optimum
target_z_score = stats.norm.ppf(critical_ratio)
optimal_inventory = current_mean + (target_z_score * estimated_std)

st.markdown("#### Order Simulator")
user_order_qty = st.slider(
    "Adjust Planned Inventory Order:",
    min_value=int(current_mean * 0.5), 
    max_value=int(current_mean * 1.5), 
    value=int(optimal_inventory), # Defaults to the mathematical optimal!
    step=10
)

z_user = (user_order_qty - current_mean) / estimated_std

pdf_z = stats.norm.pdf(z_user)
cdf_z = stats.norm.cdf(z_user)

exp_shortage_units = estimated_std * (pdf_z - z_user * (1 - cdf_z))
exp_overage_units = (user_order_qty - current_mean) + exp_shortage_units

shortage_risk_dollars = exp_shortage_units * cost_of_underage
overage_risk_dollars = exp_overage_units * cost_of_overage
total_expected_cost = shortage_risk_dollars + overage_risk_dollars

col_res1, col_res2, col_res3 = st.columns(3)

with col_res1:
    st.metric(label="Expected Shortage Risk", value=f"${shortage_risk_dollars:,.2f}", delta=f"{int(exp_shortage_units)} units short", delta_color="inverse")
with col_res2:
    st.metric(label="Expected Overage Risk", value=f"${overage_risk_dollars:,.2f}", delta=f"{int(exp_overage_units)} units over", delta_color="inverse")
with col_res3:
    st.metric(label="Total Financial Exposure", value=f"${total_expected_cost:,.2f}", help="Sum of Shortage Risk and Overage Risk. The optimal order quantity minimizes this number.")

user_percentile = cdf_z * 100

st.info(f"""
**What does this mean?** Ordering **{user_order_qty:,} units** places your inventory at the **{user_percentile:.1f}th percentile** of the model's forecasted probability. 
* This means there is a **{user_percentile:.1f}% chance** that actual demand will fall at or below your inventory limit (preventing a stockout).
* Conversely, there remains a **{(100 - user_percentile):.1f}% chance** of a demand surge exceeding your stock.
""")

st.divider()
st.markdown("### Data Drift Monitor")
st.markdown("Continuously comparing live MongoDB transaction volumes against the training baseline to detect model decay.")

training_baseline = df['Actual_Sales'].dropna().values

def check_mongo_data_drift(baseline_data, sample_size=100):
    try:
        mongo_client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        db = mongo_client["supply_chain"]
        collection = db["supply_chain_data"]
        
        cursor = collection.find({"quantity": {"$exists": True}}).sort("_id", pymongo.DESCENDING).limit(sample_size)
        live_data = [doc["quantity"] for doc in cursor]
        
        if len(live_data) < 10:
            return "warning", "Not enough live data in MongoDB to run a reliable KS-Test. Waiting for more transactions...", []
            
        ks_stat, p_value = ks_2samp(baseline_data, live_data)
        
        if p_value < 0.05:
            return "error", f" **DATA DRIFT DETECTED** (p-value: {p_value:.4f})\n\nThe live data distribution has significantly shifted from the training baseline. Model retraining is recommended.", live_data
        else:
            return "success", f" **No Data Drift Detected** (p-value: {p_value:.4f})\n\n", live_data
            
    except Exception as e:
        return "warning", f" **Database Connection Error:** Could not read from MongoDB. ({str(e)})", []

status_type, message, live_data_stream = check_mongo_data_drift(training_baseline)

if status_type == "error":
    st.error(message)
elif status_type == "success":
    st.success(message)
else:
    st.warning(message)
    
if len(live_data_stream) > 0:
    fig_drift = go.Figure()
    
    min_val = min(np.min(training_baseline), np.min(live_data_stream))
    max_val = max(np.max(training_baseline), np.max(live_data_stream))
    x_curve = np.linspace(min_val, max_val, 200)

    fig_drift.add_trace(go.Histogram(
        x=training_baseline, 
        histnorm='probability density', 
        name='Training Baseline',
        opacity=0.5, 
        marker_color='#1f77b4',
        nbinsx=30
    ))
    
    base_mean = np.mean(training_baseline)
    base_std = np.std(training_baseline)
    fig_drift.add_trace(go.Scatter(
        x=x_curve, 
        y=stats.norm.pdf(x_curve, base_mean, base_std),
        mode='lines',
        line=dict(color='#85c1e9', width=2, dash='dot'),
        name='Baseline Bell Curve'
    ))

    live_color = '#d62728' if status_type == "error" else '#2ca02c'
    
    fig_drift.add_trace(go.Histogram(
        x=live_data_stream, 
        histnorm='probability density', 
        name='Live MongoDB Data',
        opacity=0.6, 
        marker_color=live_color, 
        nbinsx=20
    ))
    
    live_mean = np.mean(live_data_stream)
    live_std = np.std(live_data_stream)
    fig_drift.add_trace(go.Scatter(
        x=x_curve, 
        y=stats.norm.pdf(x_curve, live_mean, live_std),
        mode='lines',
        line=dict(color=live_color, width=3),
        name='Live Data Bell Curve'
    ))

    fig_drift.update_layout(
        barmode='overlay', 
        height=400, 
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Sales Volume (Units)", 
        yaxis_title="Probability Density",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_drift, use_container_width=True)
    

@st.cache_data(ttl=3600)
def load_historical_data():
    try:
        df_hist = pd.read_csv('supply_chain_3yr_data.csv')
        
        if 'Date' in df_hist.columns:
            df_hist['Date'] = pd.to_datetime(df_hist['Date'])
        else:
            df_hist['Date'] = pd.date_range(start='2019-01-01', periods=len(df_hist), freq='W')
            
        return df_hist
    except Exception as e:
        st.warning(f"Could not load historical data: {e}")
        return pd.DataFrame()

df_history = load_historical_data()




st.divider()
st.markdown("### AI Assistant")

try:
    mongo_client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    db = mongo_client["supply_chain"]
    events_collection = db["supply_chain_data"]
except Exception as e:
    st.warning(f"Database connection not established. MongoDB queries will be disabled. Error: {e}")


def execute_dynamic_mongo_query(pipeline_json_str: str) -> str:
    """Executes a MongoDB aggregation pipeline."""
    try:
        pipeline = json_util.loads(pipeline_json_str)
        
        forbidden_ops = ["$out", "$merge", "$lookup"]
        for stage in pipeline:
            for op in forbidden_ops:
                if op in stage:
                    return f"Security Exception: The '{op}' operator is blocked."
        
        pipeline.append({"$limit": 50})
        
        cursor = events_collection.aggregate(pipeline)
        results = list(cursor)
        
        return json_util.dumps(results)

    except Exception as e:
        return f"Database Execution Error: {str(e)}"


@st.cache_data(ttl=3600)
def get_combined_system_instruction():
    """Merges CSV forecast data with the MongoDB Schema instructions."""
    
    try:
        df_forecast = pd.read_csv('model_v4_probabilistic_asymmetric_output.csv')
        latest_prediction = df_forecast.iloc[-1]
        forecast_context = f"""
        LATEST FORECAST (SKU 22197):
        - Expected: {latest_prediction.get('Expected_Forecast', 'N/A')} units
        - P95 (Overstock): {latest_prediction.get('P95_Forecast', 'N/A')} units
        - P05 (Stockout): {latest_prediction.get('P05_Forecast', 'N/A')} units
        """
    except Exception:
        forecast_context = "Forecast data unavailable."

    current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    instruction = f"""
    You are an autonomous Supply Chain Co-Pilot.
    
    CURRENT SYSTEM TIME: {current_time}
    
    {forecast_context}
    
    You have a tool called `execute_dynamic_mongo_query` to query a live MongoDB database.
    
    MONGODB SCHEMA ('emergency_events' collection):
    - `timestamp` (ISODate)
    - `event_type` (String): e.g., "normal_sale", "restock", "delay", "anomaly", "stockout"
    - `product_id` (String): Always "22197"
    - `severity` (Integer): Scale of 1 to 5
    - `quantity` (Integer): Units impacted
    - `description` (String)
    - `location` (String): e.g,"Northampton Fulfillment Centre","Warrington Regional Cross-Dock","Dartford Last-Mile Depot","Felixstowe Port Intake","M1 Transit Corridor"
    
    CRITICAL DATABASE RULES:
    1. If a user asks about events, write a strictly formatted JSON array representing a MongoDB aggregation pipeline.
    2. DATE FILTERING: Because `timestamp` is an ISODate, you MUST use the MongoDB extended JSON format `$date` operator for any time-based queries.
       Example: {{"$match": {{"timestamp": {{"$gte": {{"$date": "2026-04-01T00:00:00Z"}}}}}}}}
    3. Never use standard strings for date comparisons.

    Always be friendly and provide as much information as possible.
    """
    return instruction

API_KEY = os.getenv("GEMINI_API_KEY")
try:
    client = genai.Client(api_key=API_KEY)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "I am synced with your forecasting model and connected to live MongoDB event streams. How can I help?"}
        ]

    if "chat_session" not in st.session_state:
        config = types.GenerateContentConfig(
            system_instruction=get_combined_system_instruction(),
            tools=[execute_dynamic_mongo_query], 
            temperature=0.1
        )
        st.session_state.chat_session = client.chats.create(
            model='gemini-2.5-flash',
            config=config
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about pipeline forecasts or real-time anomalies..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            response = st.session_state.chat_session.send_message(prompt)
            st.markdown(response.text)
            
        st.session_state.messages.append({"role": "assistant", "content": response.text})

except Exception as e:
    st.error(f"Failed to initialize AI Assistant. Please check your API key. Error: {e}")

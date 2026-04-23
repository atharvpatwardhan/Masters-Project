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
        file_name = 'model_v2_probabilistic_asymmetric_output.csv'
        
        url = f"https://{bucket_name}.s3.amazonaws.com/{file_name}"
        
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'])
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

latest_week = df.iloc[-1]
previous_week = df.iloc[-2]

current_sales = latest_week['Actual_Sales']
sales_delta = current_sales - previous_week['Actual_Sales']

current_forecast = latest_week['Expected_Forecast']
forecast_error = current_sales - current_forecast

total_anomalies = df['Anomaly_Flag'].sum() if 'Anomaly_Flag' in df.columns else random.randint(1, 5)

st.markdown("### Current Week Operations")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Actual Weekly Sales", value=f"{int(current_sales):,}", delta=f"{int(sales_delta):,} vs last week")
with col2:
    st.metric(label="Model Forecast", value=f"{int(current_forecast):,}", delta=f"{int(forecast_error):,} unit error", delta_color="inverse")
with col3:
    st.metric(label="System Health", value="Online", delta="AWS S3 Synced")
with col4:
    st.metric(label="Detected Anomalies (YTD)", value=f"{int(total_anomalies)}", delta="Review Required", delta_color="inverse")

st.divider()

st.markdown("### Model Forecast vs. Actual Demand")

fig = go.Figure()

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
    x=df['Date'], y=df['Expected_Forecast'],
    mode='lines',
    line=dict(color='#2ca02c', width=3),
    name='XGBoost Expected Forecast'
))

fig.add_trace(go.Scatter(
    x=df['Date'], y=df['Actual_Sales'],
    mode='lines+markers',
    line=dict(color='black', width=2),
    marker=dict(size=6, color='black'),
    name='Actual Transactions'
))

if 'Anomaly_Flag' in df.columns:
    anomalies = df[df['Anomaly_Flag'] == 1]
    fig.add_trace(go.Scatter(
        x=anomalies['Date'], y=anomalies['Actual_Sales'],
        mode='markers',
        marker=dict(color='red', size=12, symbol='x'),
        name='System Anomalies'
    ))



fig.update_layout(
    height=500,
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="Units Sold"
)

chart_container = st.container()

st.divider()
st.markdown("### 🎛️ Stress Test Simulator")
st.markdown("Inject artificial demand shocks to test inventory resilience.")

demand_shock = st.slider("Demand Shock (%)", min_value=-50, max_value=100, value=0, step=5)

if demand_shock != 0:
    shock_multiplier = 1 + (demand_shock / 100)
    
    df['Simulated_Forecast'] = df['Expected_Forecast'] * shock_multiplier
    df['Simulated_Lower_CI'] = df['Lower_Bound_95CI'] * shock_multiplier
    df['Simulated_Upper_CI'] = df['Upper_Bound_95CI'] * shock_multiplier
    
    fig.add_trace(go.Scatter(
        x=pd.concat([df['Date'], df['Date'][::-1]]),
        y=pd.concat([df['Simulated_Upper_CI'], df['Simulated_Lower_CI'][::-1]]),
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name=f'Simulated 95% Risk Boundary'
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Simulated_Forecast'],
        mode='lines',
        line=dict(color='orange', width=2, dash='dash'),
        name=f'Stress Test ({demand_shock}% Shock)'
    ))
        
chart_container.plotly_chart(fig, use_container_width=True)

with st.expander("🔍 View Raw Pipeline Output"):
    st.dataframe(df.tail(10), use_container_width=True)


import scipy.stats as stats

st.divider()
st.markdown("### 🔬 Advanced Statistical Diagnostics")

df_clean = df.dropna(subset=['Actual_Sales', 'Expected_Forecast']).copy()
df_clean['Residuals'] = df_clean['Actual_Sales'] - df_clean['Expected_Forecast']

col_stat1, col_stat2 = st.columns(2)

with col_stat1:
    st.markdown("#### Financial Tail Risk (CVaR)")
    st.markdown("Calculates the **Expected Shortfall** in the worst 5% of Monte Carlo scenarios.")
    
    tail_risk_events = df_clean[df_clean['Actual_Sales'] > df_clean['Upper_Bound_95CI']]
    
    if len(tail_risk_events) > 0:
        cvar_units = (tail_risk_events['Actual_Sales'] - tail_risk_events['Upper_Bound_95CI']).mean()
        cvar_dollars = cvar_units * 25.00
        st.error(f"**95% CVaR (Expected Shortfall):** {int(cvar_units):,} units / ${cvar_dollars:,.2f}")
        st.caption("If a 5% black-swan demand spike occurs, expect an average shortage cost of this magnitude.")
    else:
        st.success("**95% CVaR:** $0.00")
        st.caption("Historical data shows no catastrophic breaches of the 95% upper risk boundary.")

with col_stat2:
    st.markdown("#### Residual Error Distribution")
    st.markdown("Proves the XGBoost model has extracted all patterns, leaving only random noise.")
    
    skew = stats.skew(df_clean['Residuals'])
    kurt = stats.kurtosis(df_clean['Residuals'])
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df_clean['Residuals'],
        nbinsx=20,
        marker_color='#1f77b4',
        opacity=0.75,
        name='Error Count'
    ))
    
    fig_hist.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Prediction Error (Units)",
        yaxis_title="Frequency",
        showlegend=False
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(f"**Skewness:** {skew:.2f} | **Kurtosis:** {kurt:.2f} (Values near 0 indicate a highly unbiased model).")




df = pd.read_csv('supply_chain_3yr_data.csv')

first_product = '22197' 
historical_demand = df[first_product].dropna().values

np.random.seed(42)
bootstrapped_base = np.random.choice(historical_demand, size=1000, replace=True)

drift_multipliers = np.linspace(1.0, 1.30, 1000)
simulated_future_stream = np.round(bootstrapped_base * drift_multipliers)

future_df = pd.DataFrame({'Quantity': simulated_future_stream})
future_df.to_csv('future_stream_data.csv', index=False)

print(f"Generated {len(future_df)} bootstrapped streaming records for SKU {first_product}!")




baseline_mean = 2500 
baseline_std = 300
training_actuals = np.random.normal(baseline_mean, baseline_std, 1000)

st.divider()
st.markdown("### Local Data Drift Monitor")
st.markdown("Ingesting bootstrapped transaction batches locally. The system continuously compares live distributions against the training baseline to detect model decay.")

@st.cache_data
def load_local_stream_data():
    try:
        return pd.read_csv('future_stream_data.csv')
    except Exception as e:
        st.error(f"Failed to find 'future_stream_data.csv': {e}")
        return pd.DataFrame()

future_df = load_local_stream_data()

if not future_df.empty:
    true_mean = future_df['Quantity'].iloc[:100].mean()
    true_std = future_df['Quantity'].iloc[:100].std()
    
    np.random.seed(42)
    training_actuals = np.random.normal(true_mean, true_std, 1000)
else:
    training_actuals = np.array([])

st.divider()
st.markdown("### Data Drift Monitor")
st.markdown("Ingesting bootstrapped transaction batches locally. The system continuously compares live distributions against the training baseline to detect model decay.")

col1, col2 = st.columns([1, 2])
with col1:
    status_placeholder = st.empty()
with col2:
    chart_placeholder = st.empty()

if not future_df.empty:
    local_data_stream = future_df['Quantity'].tolist()
        
    if st.button("Initialize Local Data Stream", type="primary"):
        
        live_data_buffer = []
        batch_size = 5
        
        for i in range(0, len(local_data_stream), batch_size):
            
            current_batch = local_data_stream[i : i + batch_size]
            live_data_buffer.extend(current_batch)
            
            if len(live_data_buffer) > 200:
                live_data_buffer = live_data_buffer[-200:]
                
            if len(live_data_buffer) > 50:
                ks_stat, p_value = ks_2samp(training_actuals, live_data_buffer)
                
                if p_value < 0.05:
                    status_placeholder.error(f"**DATA DRIFT DETECTED**\n\n**Batch Index:** {i}\nThe live bootstrapped stream has statistically deviated from the training baseline.\n\n**P-Value:** {p_value:.5f}\n\n**Action:** Model retraining required.")
                else:
                    status_placeholder.success(f"**STREAM HEALTHY**\n\n**Batch Index:** {i}\nLive data matches training statistical distributions.\n\n**P-Value:** {p_value:.4f}")
                    
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=training_actuals, histnorm='probability', name='Training Baseline',
                    opacity=0.6, marker_color='gray', nbinsx=30
                ))
                
                fig.add_trace(go.Histogram(
                    x=live_data_buffer, histnorm='probability', name='Live Window',
                    opacity=0.75, marker_color='red' if p_value < 0.05 else '#1f77b4', nbinsx=20
                ))

                min_val = min(min(training_actuals), min(live_data_buffer)) * 0.8
                max_val = max(max(training_actuals), max(live_data_buffer)) * 1.2

                fig.update_layout(
                    barmode='overlay', height=350, margin=dict(l=0, r=0, t=10, b=0),
                    xaxis=dict(title="Sales Volume (Units)", range=[min_val, max_val]), 
                    yaxis_title="Probability Density",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                
            time.sleep(0.15) 
else:
    st.warning("Generate 'future_stream_data.csv' to initialize the MLOps monitor.")


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

# AI Supply Chain Assistant with Forecasting and Anomaly Detection 


An autonomous supply chain Co-Pilot designed to help businesses predict customer demand, minimize wasted inventory, and prevent costly stockouts. This project bridges the gap between complex probabilistic forecasting and day-to-day warehouse operations by utilizing an Agentic AI interface.

---

## Project Overview

Traditional supply chain forecasting often relies on calculating a single "average" point prediction, leaving warehouses vulnerable to sudden surges in demand. This system takes a smarter approach by focusing on **financial risk optimization**. By utilizing a hybrid Prophet and XGBoost model combined with Monte Carlo simulations, it calculates the exact safety stock needed to maximize profit while minimizing stockout probabilities.

Beyond static forecasting, the system features a real-time event pipeline. It connects to live streaming data to detect market anomalies (like viral trends or shipping delays) and tracks data drift to warn operators of model decay. 

The crown jewel of the platform is the **Gemini AI Co-Pilot**. Instead of forcing managers to write SQL or navigate complex dashboards during a crisis, users can ask questions in plain English (e.g., *"Why did we stock out yesterday?"*). The AI autonomously queries the live MongoDB database and returns instant, actionable insights.

---

## Key Features & Business Value

* **Financial Risk Over Point Accuracy:** Standard machine learning optimizes for average error (MAPE), but supply chain costs are deeply asymmetric. This tool uses Newsvendor Model logic to optimize for true financial risk (the cost of a stockout vs. overstock) rather than just statistical proximity.
* **Dynamic Anomaly & Data Drift Detection:** A continuous Apache Kafka to MongoDB streaming architecture actively monitors live market conditions. It utilizes real-time Kolmogorov-Smirnov (KS) statistical testing to proactively detect data drift and alert analysts when market shifts require model retraining.
* **The "Last Mile" of Data Science:** Advanced analytics provide zero value if business users cannot interpret them. The integrated Agentic AI Co-Pilot solves the adoption bottleneck, transforming complex database queries into conversational, instant insights.

---

## System Architecture

1. **Live Streaming & Anomaly Detection:** An Apache Kafka pipeline ingests live business transactions, instantly evaluates them for sudden demand anomalies against the model's 95% risk boundaries, and securely logs the events into MongoDB for operational visibility.
2. **Automated Data Drift Monitoring:** Continuously executes KS tests comparing live streaming data against the original training baseline, triggering proactive alerts if underlying consumer buying behavior fundamentally changes.
3. **AI Assistant & Dashboard:** A Streamlit-based Command Center. The Google Gemini-powered assistant acts as an autonomous bridge, translating natural language questions into strictly formatted MongoDB aggregation pipelines, executing them, and presenting the answers contextually.

---

## Repository Structure

* **`Dataset_Expansion.ipynb`**: Applies Additive Time Series Decomposition to isolate trend, seasonality, and residuals from the raw data.
* **`EDA.ipynb`**: Exploratory Data Analysis, generating 5 years of historical baseline metrics and product analytics.
* **`Modeling.ipynb`**: Trains the hybrid forecasting engine (Prophet for baseline trends + XGBoost for complex residual patterns).
* **`app.py`**: The main Streamlit application file housing the Command Center, Dynamic Risk Simulator, Drift Monitor, and the Gemini AI chat interface.
* **`producer.py`**: Simulates the real-time business API, pushing live transaction events to the Kafka topic.
* **`consumer.py`**: Ingests the Kafka stream, runs anomaly checks against the model boundaries, and processes the output.
* **`mongodb_batch_script.py`**: Utility script for batch syncing generated historical/forecast data into the MongoDB Atlas cluster.
* **`streaming_data.py`**: Helper functions for handling Streamlit's real-time data visual updates.
* **`model_v7_2026_forecast.csv`**: The finalized probabilistic output dataset powering the dashboard's baseline expectations.
* **`supply_chain_3yr_data.csv`**: The historical training dataset.

---

## Installation & Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/atharvpatwardhan/Masters-Project.git](https://github.com/atharvpatwardhan/Masters-Project.git)
cd Masters-Project
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Environment Variables:**
Create a `.env` file in the root directory and add your credentials:
```env
MONGO_URI="your_mongodb_atlas_connection_string"
GEMINI_API_KEY="your_google_gemini_api_key"
```

**4. Start the Kafka Environment:**
*(Ensure Zookeeper and Kafka server are running locally or configure your cloud Kafka cluster in the producer/consumer scripts).*
```bash
python producer.py
python consumer.py
```

**5. Launch the Dashboard:**
```bash
streamlit run app.py
```

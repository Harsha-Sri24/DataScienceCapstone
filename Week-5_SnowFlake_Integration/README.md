
# 🌤️ WeatherTwin Dashboard

An interactive Streamlit dashboard designed to visualize historical weather patterns directly from Snowflake. This project leverages **Snowpark** to perform in-database data processing, enabling high performance even with large datasets.

## 🏗️ Architecture

The application follows a **data-app decoupled architecture**:

* **Data Layer:** Snowflake stores raw and aggregated weather data. Snowpark handles transformations.
* **Application Layer:** Streamlit dashboards (local or cloud) query and visualize the data.
* **UI Layer:** Interactive charts, filters, and metrics delivered via Streamlit.

---

## 👥 Contributors & Workflows

This project supports **two development workflows**, allowing flexibility for local and cloud-based development.

**1. Harsha Sri Neeriganti - Cloud-Native Developer (Direct in Snowflake)**

  **Environment:** Streamlit app inside Snowsight.
  
  **Authentication:** Automatic using get_active_session().
  
  **Advantage:** No setup; live connection to Snowflake.
  
  **Responsibilities & Contributions:**

    Uploaded raw weather data directly to Snowflake via CSV ingestion.
    
    Developed the multi-city interactive dashboard, enabling selection and comparison of multiple cities.
    
    Implemented real-time metrics calculations (Max/Min temperatures, Precipitation) using Snowpark in-database computations.
    
    Optimized queries for performance, ensuring the dashboard handles large datasets efficiently.
    
    Integrated time-series visualizations and dynamic filters using Streamlit and Plotly.

**2. Sayush Maharjan - Local Developer (Remote Connection)**

  **Environment:** Local IDE (VS Code / PyCharm).
  
  **Authentication:** Manual; uses creds.json or secrets.py.
  
  **Advantage:** Full control, faster iteration, and local debugging tools.
  
  **Responsibilities & Contributions:**
  
    Connected to Snowflake to fetch and transform data, creating local pipelines for testing.
    
    Developed the local Streamlit dashboard for offline testing and iteration.
    
    Implemented hybrid code support, ensuring the same logic works both locally and in the Snowflake cloud.
    
    Added interactive charts and dynamic filters for quick exploration of historical weather trends.
    
    Assisted with data validation and cleaning, ensuring consistent results across environments.
  
  💡 Together, both workflows allow seamless development: the cloud version offers zero-setup deployment, while the local version provides flexibility for testing and feature iteration.
  
  ---

## 🚀 Setup & Installation

### For Local Contributors

1. **Clone the repository:**

```bash
git clone https://github.com/your-repo/weather-dashboard.git
cd weather-dashboard
```

2. **Install dependencies:**

```bash
pip install snowflake-snowpark-python streamlit pandas plotly
```

3. **Configure credentials:**
   Create a `creds.json` file (never commit this file):

```json
{
  "account": "your_org-your_account",
  "user": "your_username",
  "password": "your_password",
  "role": "your_role",
  "warehouse": "your_wh",
  "database": "WEATHER_DB",
  "schema": "PUBLIC"
}
```

### For Cloud Contributors

1. Open **Snowsight** → **Streamlit**.
2. Create a new app.
3. Copy `app_snowflake.py` contents (or your main `streamlit_app.py`) into the editor.
4. Add required packages via the **Packages** dropdown.

---

## 🛠️ Hybrid Code Support

To support both local and cloud environments in a single codebase:

```python
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session

def get_session():
    try:
        # Cloud environment
        return get_active_session()
    except:
        # Local environment
        import json
        with open('creds.json') as f:
            connection_parameters = json.load(f)
        return Session.builder.configs(connection_parameters).create()

session = get_session()
```

---

## 📂 Project Structure

```
weather-dashboard/
│
├─ load_weather_csv.py
│
├─ snowflake_app.py            # Streamlit app for cloud deployment in Snowflake
├─ app_bert.py                # Streamlit app for local development
├─ snowflake_client.py            # Shared core logic (optional if using hybrid code)
│
├─ requirements.txt            # Local dependencies
├─ environment.yml             # Cloud environment dependencies
```

**Notes:**

* `app_snowflake.py` → For **cloud deployment** inside Snowsight.
* `app_local.py` → For **local Streamlit testing** and debugging.
* `ingestion/` → Handles initial data ingestion from CSV or external sources into Snowflake.

---

## 📊 Features

* **Dynamic Filtering:** Searchable dropdown populated with `NAME` values from the `WEATHER_FULL` table.
* **Live Metrics:** Calculates Max/Min temperatures and Precipitation in real time.
* **Historical Trends:** Time-series charts using Streamlit and Plotly.
* **Environment-Agnostic:** Single codebase supports both local and cloud workflows.
-

Dashboard demo video link: https://drive.google.com/file/d/1VK4R-Iro2UsWvcMJRorbFTdT3lqSHYHk/view?usp=drive_link 

Cloud Snowflake dashboard link (Snowsight): https://app.snowflake.com/sfedu02/dcb73175/#/streamlit-apps/WEATHER_TWIN_DB.PUBLIC.LMTE323F0FBAR_NK 


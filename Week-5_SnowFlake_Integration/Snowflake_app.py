import streamlit as st
import pandas as pd
import snowflake.snowpark.functions as F
from snowflake.snowpark.context import get_active_session

# -------------------------------------------------
# 1. PAGE SETUP
# -------------------------------------------------
st.set_page_config(
    page_title="WeatherTwin Explorer",
    layout="wide",
    page_icon="🌤️"
)

st.title("🌤️ WeatherTwin Dashboard")
st.markdown("Interactive multi-city weather analytics powered by Snowflake.")

# -------------------------------------------------
# 2. GET ACTIVE SESSION (Snowflake Projects only)
# -------------------------------------------------
try:
    session = get_active_session()
except Exception:
    st.error("No active Snowflake session found. Run inside Snowflake Projects.")
    st.stop()

# -------------------------------------------------
# 3. CACHE: GET UNIQUE CITIES
# -------------------------------------------------
@st.cache_data
def get_unique_cities():
    df = (
        session.table("WEATHER_FULL")
        .select("NAME")
        .distinct()
        .to_pandas()
    )
    return sorted(df["NAME"].tolist())

cities = get_unique_cities()

# -------------------------------------------------
# 4. SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("Filter Settings")

selected_cities = st.sidebar.multiselect(
    "📍 Select City/Cities",
    options=cities,
    default=cities[:1] if cities else []
)

metrics_options = ["EMXT", "EMNT", "PRCP", "AWND"]

selected_metrics = st.sidebar.multiselect(
    "📊 Select Metrics",
    options=metrics_options,
    default=["EMXT", "EMNT"]
)

# -------------------------------------------------
# 5. LOAD FILTERED DATA
# -------------------------------------------------
if selected_cities:

    weather_df = (
        session.table("WEATHER_FULL")
        .filter(F.col("NAME").isin(selected_cities))
        .sort(F.col("DATE").asc())
        .to_pandas()
    )

    if weather_df.empty:
        st.warning("No data found for selected cities.")
        st.stop()

    # Convert DATE column
    weather_df["DATE"] = pd.to_datetime(weather_df["DATE"])

    # -------------------------------------------------
    # 6. DATE RANGE FILTER
    # -------------------------------------------------
    min_date = weather_df["DATE"].min()
    max_date = weather_df["DATE"].max()

    start_date, end_date = st.sidebar.date_input(
        "📅 Select Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    weather_df = weather_df[
        (weather_df["DATE"] >= pd.to_datetime(start_date)) &
        (weather_df["DATE"] <= pd.to_datetime(end_date))
    ]

    # -------------------------------------------------
    # 7. KPI SECTION (Latest Per City)
    # -------------------------------------------------
    st.subheader("📈 Latest Metrics Per City")

    for city in selected_cities:
        city_data = weather_df[weather_df["NAME"] == city]
        if not city_data.empty:
            latest = city_data.sort_values("DATE", ascending=False).iloc[0]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("City", city)
            col2.metric("Max Temp (EMXT)", f"{latest['EMXT']}°")
            col3.metric("Min Temp (EMNT)", f"{latest['EMNT']}°")
            col4.metric("Precipitation (PRCP)", f"{latest['PRCP']}")

    st.divider()

    # -------------------------------------------------
    # 8. WEATHER TRENDS (Snowflake-safe charts)
    # -------------------------------------------------
    st.subheader("📊 Weather Trends")

    if selected_metrics:
        for metric in selected_metrics:

            st.markdown(f"### {metric}")

            pivot_df = (
                weather_df
                .pivot(index="DATE", columns="NAME", values=metric)
                .sort_index()
            )

            st.line_chart(pivot_df, use_container_width=True)

    else:
        st.info("Select at least one metric to display trends.")

    # -------------------------------------------------
    # 9. CITY RANKING SECTION
    # -------------------------------------------------
    st.subheader("🏆 City Rankings (Average Values)")

    ranking_df = (
        weather_df
        .groupby("NAME")[selected_metrics]
        .mean()
        .reset_index()
    )

    if not ranking_df.empty:
        st.dataframe(ranking_df, use_container_width=True)

    # -------------------------------------------------
    # 10. RAW DATA TABLE
    # -------------------------------------------------
    with st.expander("📄 View Full Filtered Dataset"):
        st.write(f"Showing {len(weather_df)} records.")
        st.dataframe(weather_df, use_container_width=True)

else:
    st.info("Please select at least one city from the sidebar.")

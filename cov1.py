import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# --------------------------------
# LOAD DATA
# --------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"d:\datasets\Covid\full_grouped.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

st.title("📊 COVID‑19 Dashboard — Global Analysis")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["📈 Time Series", "🌍 Country Comparison", "📉 Projection & Indicators"]
)

# ==========================================
# PAGE 1 — TIME SERIES
# ==========================================
if page == "📈 Time Series":

    st.header("📈 Evolution des cas / décès dans le monde")

    metric = st.selectbox("Choose metric:", ["Confirmed", "Deaths", "Recovered", "Active"])

    df_world = df.groupby("Date")[metric].sum().reset_index()

    fig = px.line(df_world, x="Date", y=metric, title=f"Global {metric} Over Time")
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# PAGE 2 — COUNTRY COMPARISON
# ==========================================
elif page == "🌍 Country Comparison":

    st.header("🌍 Comparaison entre les pays")

    country_list = sorted(df["Country/Region"].unique())
    selected_countries = st.multiselect("Select countries:", country_list, ["France", "Italy", "Spain"])

    metric = st.selectbox("Metric:", ["Confirmed", "Deaths", "Recovered", "Active"])

    df_selected = df[df["Country/Region"].isin(selected_countries)]

    fig = px.line(
        df_selected,
        x="Date",
        y=metric,
        color="Country/Region",
        title=f"{metric} comparison between countries"
    )
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# PAGE 3 — PROJECTION
# ==========================================
elif page == "📉 Projection & Indicators":

    st.header("📉 Simple Projection (Linear Trend)")

    metric = st.selectbox("Metric to project:", ["Confirmed", "Deaths"])

    df_world = df.groupby("Date")[metric].sum().reset_index()
    df_world["day"] = np.arange(len(df_world))

    # Simple linear model
    model = LinearRegression()
    model.fit(df_world[["day"]], df_world[metric])

    future_days = 15
    future_idx = np.arange(len(df_world), len(df_world) + future_days).reshape(-1, 1)
    predictions = model.predict(future_idx)

    # Create projection dataframe
    df_future = pd.DataFrame({
        "Date": pd.date_range(df_world["Date"].iloc[-1], periods=future_days + 1, freq="D")[1:],
        metric: predictions
    })

    # Plot
    fig = px.line(df_world, x="Date", y=metric, title=f"{metric} Projection (next {future_days} days)")
    fig.add_scatter(x=df_future["Date"], y=df_future[metric], mode="lines", name="Prediction")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Key Indicators")
    st.write(f"**Last real value ({metric})**: {df_world[metric].iloc[-1]:,.0f}")
    st.write(f"**Predicted value in {future_days} days**: {predictions[-1]:,.0f}")
    st.write(f"**Daily increase rate (approx.)**: {model.coef_[0]:,.0f} per day")

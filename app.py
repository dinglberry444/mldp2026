# app.py
# Streamlit web app for predicting Singapore COE premiums
# Machine Learning for Developers (CAI2C08) - End of Semester Project

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="COE Premium Predictor", page_icon="📊", layout="wide")

# a little custom styling to make it look cleaner
st.markdown(
    """
    <style>
    .big-title { font-size: 40px; font-weight: 800; margin-bottom: 0; }
    .subtitle { color: #9aa0a6; font-size: 16px; margin-top: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Load the saved model, feature columns, and historical data
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("best_coe_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_columns.pkl", "rb") as f:
        columns = pickle.load(f)
    return model, columns

@st.cache_data
def load_history():
    # used to compare the prediction against real historical premiums per category
    df = pd.read_csv("COEBiddingResultsPrices.csv")
    return df.groupby("vehicle_class")["premium"].agg(["mean", "min", "max"])

model, model_columns = load_model()
cat_stats = load_history()

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown('<p class="big-title">COE Premium Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Estimate the COE premium before the bidding result is announced &mdash; '
    'a pricing tool for used-car dealers.</p>',
    unsafe_allow_html=True,
)
st.write("")

# ------------------------------------------------------------------
# Sidebar - user inputs
# ------------------------------------------------------------------
st.sidebar.header("Bidding round details")

vehicle_class = st.sidebar.selectbox(
    "Vehicle class",
    ["Category A", "Category B", "Category C", "Category D", "Category E"],
)
bidding_no = st.sidebar.radio("Bidding exercise", [1, 2], horizontal=True)
quota = st.sidebar.slider("Quota (COEs available)", 40, 2300, 1300, step=10)
bids_received = st.sidebar.slider("Bids received (demand)", 50, 6000, 2400, step=10)

col_y, col_m = st.sidebar.columns(2)
year = col_y.number_input("Year", min_value=2010, max_value=2030, value=2026)
month_num = col_m.number_input("Month", min_value=1, max_value=12, value=7)

predict = st.sidebar.button("Predict premium", type="primary", use_container_width=True)

# ------------------------------------------------------------------
# Main area - prediction and results
# ------------------------------------------------------------------
if predict:

    # ---- input validation ----
    if quota <= 0:
        st.error("Quota must be greater than 0 - a bidding round can't have zero COEs available.")
        st.stop()
    if bids_received <= 0:
        st.error("Bids received must be greater than 0 - please enter the number of bids.")
        st.stop()

    ratio = bids_received / quota
    if bids_received < quota:
        st.warning(
            "Bids received is below the quota (under-subscribed). This is unusual for COE "
            "bidding - please double-check your inputs before trusting the prediction."
        )

    # ---- build the feature row in the same format as training ----
    input_dict = {
        "bidding_no": bidding_no,
        "quota": quota,
        "bids_received": bids_received,
        "year": year,
        "month_num": month_num,
        "oversubscription_ratio": ratio,
        "vehicle_class_Category B": 1 if vehicle_class == "Category B" else 0,
        "vehicle_class_Category C": 1 if vehicle_class == "Category C" else 0,
        "vehicle_class_Category D": 1 if vehicle_class == "Category D" else 0,
        "vehicle_class_Category E": 1 if vehicle_class == "Category E" else 0,
    }
    input_df = pd.DataFrame([input_dict])[model_columns]

    # ---- predict ----
    with st.spinner("Crunching the numbers..."):
        prediction = model.predict(input_df)[0]

    cat_avg = cat_stats.loc[vehicle_class, "mean"]
    cat_max = cat_stats.loc[vehicle_class, "max"]
    delta = prediction - cat_avg

    st.success("Prediction complete")

    # ---- headline metrics ----
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Predicted premium",
        "${:,.0f}".format(prediction),
        "{:+,.0f} vs {} average".format(delta, vehicle_class),
    )
    demand_label = "hot demand" if ratio >= 2 else ("moderate demand" if ratio >= 1 else "under-subscribed")
    c2.metric("Oversubscription ratio", "{:.2f}x".format(ratio), demand_label, delta_color="off")
    c3.metric("{} average (historical)".format(vehicle_class), "${:,.0f}".format(cat_avg))

    # ---- comparison chart ----
    st.write("")
    st.subheader("How this prediction compares")
    chart_df = pd.DataFrame(
        {"premium ($)": [prediction, cat_avg, cat_max]},
        index=["This prediction", "{} average".format(vehicle_class), "{} highest ever".format(vehicle_class)],
    )
    st.bar_chart(chart_df)

    # ---- explanation ----
    with st.expander("What do these inputs mean?"):
        st.write("- **Quota** - number of COEs available that round (the supply).")
        st.write("- **Bids received** - number of bids submitted that round (the demand).")
        st.write("- **Oversubscription ratio** - bids received divided by quota. Above 2.0 means demand is hot, which usually pushes the premium up.")

    st.caption("This is an estimate from historical patterns and should be used as a pricing guide, not a guaranteed result.")

else:
    st.info("Set the bidding round details in the sidebar on the left, then click **Predict premium**.")

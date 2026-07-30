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
st.set_page_config(page_title="COE Premium Predictor", page_icon="📊", layout="centered")

st.markdown(
    """
    <style>
    .big-title { font-size: 38px; font-weight: 800; margin-bottom: 0; }
    .subtitle { color: #9aa0a6; font-size: 16px; margin-top: 4px; margin-bottom: 10px; }
    .result-card { background: #14532d; border-radius: 14px; padding: 20px 26px;
                   text-align: center; margin: 8px 0 14px 0; }
    .result-label { color: #b7e4c7; font-size: 15px; }
    .result-value { color: #ffffff; font-size: 46px; font-weight: 800; line-height: 1.15; }
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
    # used to compare the prediction against the typical premium for that category
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
    'a quick pricing guide for used-car dealers.</p>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Sidebar - user inputs (each with a plain-language help tooltip)
# ------------------------------------------------------------------
st.sidebar.header("Enter the bidding round")
st.sidebar.caption("Fill in the details below, then click Predict premium.")

vehicle_class = st.sidebar.selectbox(
    "Vehicle class",
    ["Category A", "Category B", "Category C", "Category D", "Category E"],
    help="A and B are cars, C is larger/commercial vehicles, D is motorcycles, E is Open.",
)
bidding_no = st.sidebar.radio(
    "Bidding exercise", [1, 2], horizontal=True,
    help="COE bidding runs twice a month. 1 = first round, 2 = second round.",
)
quota = st.sidebar.slider(
    "Quota (COEs available)", 40, 2300, 1300, step=10,
    help="How many COEs are up for grabs this round - the supply.",
)
bids_received = st.sidebar.slider(
    "Bids received (demand)", 50, 6000, 2400, step=10,
    help="How many bids were submitted this round - the demand.",
)
col_y, col_m = st.sidebar.columns(2)
year = col_y.number_input("Year", min_value=2010, max_value=2030, value=2026,
                          help="The year of the bidding round.")
month_num = col_m.number_input("Month", min_value=1, max_value=12, value=7,
                               help="The month of the bidding round (1-12).")

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
    with st.spinner("Estimating the premium..."):
        prediction = model.predict(input_df)[0]

    cat_avg = cat_stats.loc[vehicle_class, "mean"]

    # ---- result card ----
    st.markdown(
        '<div class="result-card">'
        '<div class="result-label">Predicted {} premium</div>'
        '<div class="result-value">${:,.0f}</div>'
        '</div>'.format(vehicle_class, prediction),
        unsafe_allow_html=True,
    )

    # ---- plain-language explanation (replaces the chart) ----
    diff = prediction - cat_avg
    direction = "higher than" if diff >= 0 else "lower than"
    demand = "strong (hot)" if ratio >= 2 else ("normal" if ratio >= 1 else "weak (under-subscribed)")
    st.write(
        "With **{:,} bids** for **{:,} COEs**, demand is **{:.2f}x the supply** ({}). "
        "The estimated premium is **\\${:,.0f}**, which is **\\${:,.0f} {}** the typical "
        "{} premium of **\\${:,.0f}**.".format(
            bids_received, quota, ratio, demand, prediction, abs(diff), direction, vehicle_class, cat_avg
        )
    )

    # ---- two simple supporting numbers ----
    col1, col2 = st.columns(2)
    col1.metric("Oversubscription ratio", "{:.2f}x".format(ratio))
    col2.metric("Typical {} premium".format(vehicle_class), "${:,.0f}".format(cat_avg))

    st.caption(
        "Estimate based on historical bidding patterns - use as a pricing guide, not a guaranteed result."
    )

else:
    st.info("Enter the bidding round details in the panel on the left, then click **Predict premium**.")

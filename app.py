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

st.title("COE Premium Predictor")
st.write(
    "A B2B tool for used-car dealers to estimate the COE premium **before** the "
    "bidding result is announced. Enter the mid-round bidding details below and "
    "the model will predict the likely premium."
)

# ------------------------------------------------------------------
# Load the saved model and the feature column order
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("best_coe_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_columns.pkl", "rb") as f:
        columns = pickle.load(f)
    return model, columns

model, model_columns = load_model()

# ------------------------------------------------------------------
# User inputs (laid out in two columns for a clean look)
# ------------------------------------------------------------------
st.header("Bidding Round Details")

col1, col2 = st.columns(2)

with col1:
    vehicle_class = st.selectbox(
        "Vehicle Class",
        ["Category A", "Category B", "Category C", "Category D", "Category E"]
    )
    quota = st.number_input("Quota (COEs available)", min_value=0, value=1300, step=10)
    bids_received = st.number_input("Bids Received (demand)", min_value=0, value=2400, step=10)

with col2:
    bidding_no = st.selectbox("Bidding Exercise No.", [1, 2])
    year = st.number_input("Year", min_value=2010, max_value=2030, value=2026, step=1)
    month_num = st.number_input("Month (1-12)", min_value=1, max_value=12, value=7, step=1)

# ------------------------------------------------------------------
# Prediction with input validation
# ------------------------------------------------------------------
if st.button("Predict Premium"):

    # ---- Input validation: stop before predicting if data is invalid ----
    if quota <= 0:
        st.error("Quota must be greater than 0. There cannot be a bidding round with no COEs available.")
        st.stop()

    if bids_received <= 0:
        st.error("Bids Received must be greater than 0. Please enter the number of bids.")
        st.stop()

    if bids_received < quota:
        st.warning(
            "Bids Received is less than the Quota, so this round is under-subscribed. "
            "This is unusual for COE bidding - please double-check your inputs before trusting the prediction."
        )

    # ---- Build the feature row in the SAME format as training ----
    oversubscription_ratio = bids_received / quota

    input_dict = {
        "bidding_no": bidding_no,
        "quota": quota,
        "bids_received": bids_received,
        "year": year,
        "month_num": month_num,
        "oversubscription_ratio": oversubscription_ratio,
        "vehicle_class_Category B": 1 if vehicle_class == "Category B" else 0,
        "vehicle_class_Category C": 1 if vehicle_class == "Category C" else 0,
        "vehicle_class_Category D": 1 if vehicle_class == "Category D" else 0,
        "vehicle_class_Category E": 1 if vehicle_class == "Category E" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    # Make sure the columns are in the exact order the model was trained on
    input_df = input_df[model_columns]

    # ---- Predict ----
    prediction = model.predict(input_df)[0]

    st.success("Prediction complete!")
    st.metric(label="Predicted COE Premium", value="$ {:,.0f}".format(prediction))

    st.write("**Oversubscription ratio for this round:** {:.2f}".format(oversubscription_ratio))
    st.caption(
        "Note: this is an estimate from historical patterns and should be used as a "
        "pricing guide, not a guaranteed result."
    )

st.write("---")
st.caption("Model: Tuned Random Forest Regressor | Module: CAI2C08 Machine Learning for Developers")

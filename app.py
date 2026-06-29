import streamlit as st
import pandas as pd
import numpy as np
import joblib

def feature_engineering(data):
    data["area_per_story"] = data["area"] / data["stories"]
    data["bedrooms_per_story"] = data["bedrooms"] / data["stories"]
    data["bedrooms_per_area"] = data["bedrooms"] / data["area"]
    return data
 
st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# LOAD THE SAVED PIPELINE (cached so it only loads once)
@st.cache_resource
def load_model():
    return joblib.load("housing_model.pkl")

model = load_model()

Main_Road_Option = ["yes", "no"]
Guest_room_Option = ["yes", "no"]
Basement_Option = ["yes", "no"]
Hot_Water_Heating_Option = ["yes", "no"]
Air_Conditioning_Option = ["yes", "no"]
Prefarea_Option = ["yes", "no"]
Furnishingst_Option = ["furnished", "unfurnished", "semi-furnished"]

# CONSTANTS - must match what the pipeline was trained on (RAW columns only;
# the pipeline's internal feature_engineering step derives the ratio features)
cat_attribs = [
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea", "furnishingstatus"
]

num_attribs = ["area", "bedrooms", "bathrooms", "stories", "parking"]

# Header
st.title("🏠 Olawale Housing Price Predictor")
st.write(
    "Predict house price for a property using a Random Forest "
    "model trained on the Housing dataset."
)

tab1, tab2 = st.tabs(["🔢 Single Prediction", "📂 Batch Prediction (CSV)"])

# TAB 1: MANUAL SINGLE-ROW INPUT
with tab1:
    st.subheader("Enter property details")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        area = st.number_input(
            "Area (sqft)", min_value=1650, max_value=13200, value=3000, step=10
        )
        bedrooms = st.number_input(
            "Bedrooms", min_value=1, max_value=6, value=3, step=1
        )
        bathrooms = st.number_input(
            "Bathrooms", min_value=1, max_value=4, value=2, step=1
        )

    with col2:
        stories = st.number_input(
            "Stories", min_value=1, max_value=4, value=2, step=1
        )
        parking = st.number_input(
            "Parking", min_value=0, max_value=3, value=1, step=1
        )
        mainroad = st.selectbox("Mainroad", Main_Road_Option)

    with col3:
        guestroom = st.selectbox("Guestroom", Guest_room_Option)
        basement = st.selectbox("Basement", Basement_Option)
        hotwaterheating = st.selectbox("Hot Water Heating", Hot_Water_Heating_Option)

    with col4:
        airconditioning = st.selectbox("Airconditioning", Air_Conditioning_Option)
        prefarea = st.selectbox("Prefarea", Prefarea_Option)
        furnishingstatus = st.selectbox("Furnishing Status", Furnishingst_Option)

    if st.button("Predict House Value", type="primary"):
        input_df = pd.DataFrame([{
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "parking": parking,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus
        }])

        try:
            prediction = model.predict(input_df)[0]
            st.success(f"### Predicted House Price: ${prediction:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# TAB 2: BATCH PREDICTION VIA CSV UPLOAD
with tab2:
    st.subheader("Upload a CSV for batch predictions")
    st.write(
        "Your CSV must contain these raw columns (the pipeline will "
        "engineer the ratio features automatically): "
        f"`{', '.join(num_attribs + cat_attribs)}`"
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            required_cols = set(num_attribs + cat_attribs)
            missing_cols = required_cols - set(batch_df.columns)
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                predictions = model.predict(batch_df)
                results_df = batch_df.copy()
                results_df["predicted_price"] = predictions

                st.write("### Predictions")
                st.dataframe(results_df)

                csv_download = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results as CSV",
                    data=csv_download,
                    file_name="housing_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Could not process file: {e}")

# FOOTER
st.markdown("---")
st.caption(
    "Model: Random Forest Regressor with full preprocessing pipeline "
    "(feature engineering, imputation, scaling, one-hot encoding)."
)
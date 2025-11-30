import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import clean_data, get_preprocessor
from src.feature_engineering import add_features

# Page configuration
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .canceled {
        background-color: #FFE5E5;
        color: #FF4B4B;
        border: 2px solid #FF4B4B;
    }
    .not-canceled {
        background-color: #E5F5E5;
        color: #28A745;
        border: 2px solid #28A745;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/model_pipeline.joblib')
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please train the model first by running `python main.py --test`")
        return None

# Title and description
st.title("🏨 Hotel Booking Cancellation Predictor")
st.markdown("""
This application predicts whether a hotel booking will be **canceled** or **not canceled** 
using a trained XGBoost model with **94.9% ROC AUC** accuracy.
""")

# Sidebar - Model Information
with st.sidebar:
    st.header("📊 Model Information")
    st.metric("Model", "XGBoost (Tuned)")
    st.metric("ROC AUC", "94.9%")
    st.metric("Accuracy", "87.6%")
    st.metric("F1 Score", "81.8%")
    
    st.markdown("---")
    st.subheader("🎯 Top Predictors")
    st.markdown("""
    1. **Market Segment** (23%)
    2. **Special Requests** (9.6%)
    3. **Lead Time** (9.2%)
    4. **Parking Space** (6.4%)
    """)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Booking Details")
    
    # Guest information
    no_of_adults = st.number_input("Number of Adults", min_value=0, max_value=10, value=2)
    no_of_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    
    # Stay details
    no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=30, value=1)
    no_of_week_nights = st.number_input("Week Nights", min_value=0, max_value=30, value=2)
    
    # Meal and room
    type_of_meal_plan = st.selectbox("Meal Plan", 
        ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
    room_type_reserved = st.selectbox("Room Type", 
        ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", 
         "Room_Type 5", "Room_Type 6", "Room_Type 7"])

with col2:
    st.subheader("🔧 Additional Information")
    
    # Booking details
    lead_time = st.number_input("Lead Time (days before arrival)", min_value=0, max_value=500, value=50)
    avg_price_per_room = st.number_input("Average Price per Room ($)", min_value=0.0, max_value=500.0, value=100.0)
    
    # Market and preferences
    market_segment_type = st.selectbox("Market Segment", 
        ["Online", "Offline", "Corporate", "Aviation", "Complementary"])
    required_car_parking_space = st.selectbox("Parking Required?", [0, 1])
    no_of_special_requests = st.number_input("Special Requests", min_value=0, max_value=10, value=0)
    
    # Guest history
    repeated_guest = st.selectbox("Repeated Guest?", [0, 1])
    no_of_previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=50, value=0)
    no_of_previous_bookings_not_canceled = st.number_input("Previous Bookings (Not Canceled)", 
        min_value=0, max_value=100, value=0)

# Arrival date
st.subheader("📅 Arrival Date")
col_date1, col_date2, col_date3 = st.columns(3)
with col_date1:
    arrival_year = st.number_input("Year", min_value=2017, max_value=2030, value=2024)
with col_date2:
    arrival_month = st.number_input("Month", min_value=1, max_value=12, value=6)
with col_date3:
    arrival_date = st.number_input("Day", min_value=1, max_value=31, value=15)

# Predict button
st.markdown("---")
if st.button("🔮 Predict Cancellation Risk"):
    model = load_model()
    
    if model is not None:
        # Create input dataframe
        input_data = pd.DataFrame({
            'Booking_ID': ['PRED00001'],
            'no_of_adults': [no_of_adults],
            'no_of_children': [no_of_children],
            'no_of_weekend_nights': [no_of_weekend_nights],
            'no_of_week_nights': [no_of_week_nights],
            'type_of_meal_plan': [type_of_meal_plan],
            'required_car_parking_space': [required_car_parking_space],
            'room_type_reserved': [room_type_reserved],
            'lead_time': [lead_time],
            'arrival_year': [arrival_year],
            'arrival_month': [arrival_month],
            'arrival_date': [arrival_date],
            'market_segment_type': [market_segment_type],
            'repeated_guest': [repeated_guest],
            'no_of_previous_cancellations': [no_of_previous_cancellations],
            'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
            'avg_price_per_room': [avg_price_per_room],
            'no_of_special_requests': [no_of_special_requests],
            'booking_status': ['Not_Canceled']  # Dummy value
        })
        
        # Preprocess
        try:
            df_clean = clean_data(input_data)
            df_feat = add_features(df_clean)
            X = df_feat.drop(columns=['target'])
            
            # Predict
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box canceled">
                    ⚠️ HIGH RISK: Booking Likely to be CANCELED
                </div>
                """, unsafe_allow_html=True)
                
                st.error(f"**Cancellation Probability:** {probability[1]*100:.1f}%")
                
                st.markdown("### 💡 Recommended Actions:")
                st.markdown("""
                - Send confirmation email with booking details
                - Offer flexible cancellation policy
                - Provide incentives (free parking, meal upgrade)
                - Follow up 48 hours before arrival
                - Maintain waitlist for this date
                """)
            else:
                st.markdown(f"""
                <div class="prediction-box not-canceled">
                    ✅ LOW RISK: Booking Likely to be CONFIRMED
                </div>
                """, unsafe_allow_html=True)
                
                st.success(f"**Confirmation Probability:** {probability[0]*100:.1f}%")
                
                st.markdown("### 💡 Recommended Actions:")
                st.markdown("""
                - Send standard confirmation email
                - Prepare room as scheduled
                - No special intervention needed
                - Standard check-in procedures
                """)
            
            # Show probability breakdown
            col_prob1, col_prob2 = st.columns(2)
            with col_prob1:
                st.metric("Not Canceled Probability", f"{probability[0]*100:.1f}%")
            with col_prob2:
                st.metric("Canceled Probability", f"{probability[1]*100:.1f}%")
                
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Powered by XGBoost ML Model | ROC AUC: 94.9% | Accuracy: 87.6%</p>
    <p>GitHub: <a href='https://github.com/Simranjangra31/Hotel_Booking_Pipeline'>Hotel_Booking_Pipeline</a></p>
</div>
""", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# 1. LOAD ASSETS
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load('airline_model.joblib')
    scaler = joblib.load('scaler.joblib')
    encoders = joblib.load('encoders.joblib')
    return model, scaler, encoders

model, scaler, encoders = load_assets()

# ---------------------------------------------------------
# 2. UI HEADER
# ---------------------------------------------------------
st.title("✈️ Prediksi Permintaan Maskapai Penerbangan")
st.write("Aplikasi Machine Learning menggunakan Algoritma **Random Forest**")
st.markdown("---")

# ---------------------------------------------------------
# 3. USER INPUT FORM
# ---------------------------------------------------------
# Divide layout into 2 columns
col1, col2 = st.columns(2)

with col1:
    num_passengers = st.number_input("Jumlah Penumpang", min_value=1, max_value=10, value=1)
    sales_channel = st.selectbox("Sales Channel", encoders['sales_channel'].classes_)
    trip_type = st.selectbox("Tipe Perjalanan", encoders['trip_type'].classes_)
    purchase_lead = st.number_input("Jarak Waktu Booking (Hari)", min_value=0, max_value=500, value=30)
    length_of_stay = st.number_input("Durasi Menginap (Hari)", min_value=0, max_value=365, value=5)

with col2:
    flight_hour = st.slider("Jam Penerbangan", 0, 23, 12)
    flight_day = st.selectbox("Hari Penerbangan", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    # Limiting Route and Origin for demo performance (using the classes from encoder)
    route = st.selectbox("Rute Penerbangan", encoders['route'].classes_)
    booking_origin = st.selectbox("Asal Booking (Negara)", encoders['booking_origin'].classes_)
    flight_duration = st.number_input("Durasi Penerbangan (Jam)", min_value=1.0, max_value=24.0, value=5.0)

st.markdown("---")
st.subheader("Opsi Tambahan")
c1, c2, c3 = st.columns(3)
with c1:
    wants_extra_baggage = st.checkbox("Extra Baggage", value=False)
with c2:
    wants_preferred_seat = st.checkbox("Preferred Seat", value=False)
with c3:
    wants_in_flight_meals = st.checkbox("In-flight Meals", value=False)

# ---------------------------------------------------------
# 4. PREDICTION LOGIC
# ---------------------------------------------------------
if st.button("Prediksi Sekarang", type="primary"):
    
    # A. Map Manual Inputs (Day of Week)
    day_mapping = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
    flight_day_num = day_mapping[flight_day]

    # B. Convert Booleans to Int
    baggage = 1 if wants_extra_baggage else 0
    seat = 1 if wants_preferred_seat else 0
    meals = 1 if wants_in_flight_meals else 0

    # C. Create DataFrame for Input
    input_data = pd.DataFrame({
        'num_passengers': [num_passengers],
        'sales_channel': [sales_channel],
        'trip_type': [trip_type],
        'purchase_lead': [purchase_lead],
        'length_of_stay': [length_of_stay],
        'flight_hour': [flight_hour],
        'flight_day': [flight_day_num],
        'route': [route],
        'booking_origin': [booking_origin],
        'wants_extra_baggage': [baggage],
        'wants_preferred_seat': [seat],
        'wants_in_flight_meals': [meals],
        'flight_duration': [flight_duration]
    })

    try:
        # D. Encode Categorical
        input_data['sales_channel'] = encoders['sales_channel'].transform(input_data['sales_channel'])
        input_data['trip_type'] = encoders['trip_type'].transform(input_data['trip_type'])
        input_data['route'] = encoders['route'].transform(input_data['route'])
        input_data['booking_origin'] = encoders['booking_origin'].transform(input_data['booking_origin'])

        # E. Scale Numerical (Important: Must convert to float first)
        num_cols = ['purchase_lead', 'length_of_stay', 'flight_duration']
        input_data[num_cols] = input_data[num_cols].astype('float64')
        input_data[num_cols] = scaler.transform(input_data[num_cols])

        # F. Predict
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        # G. Display Result
        st.markdown("### Hasil Prediksi:")
        if prediction[0] == 1:
            st.success(f"✅ Booking Berhasil! (Probabilitas: {probability:.2%})")
        else:
            st.error(f"❌ Booking Tidak Selesai. (Probabilitas Booking: {probability:.2%})")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan input: {e}")

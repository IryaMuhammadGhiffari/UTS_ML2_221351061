import streamlit as st
import tensorflow  as tf # type: ignore
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="obesity-prediction.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Obesity Prediction")
st.write("Mendiagnosa Obesitas.")

# Form input pengguna
age = st.number_input("Umur", min_value=0.0, max_value=90.0, value=20.0)
height = st.number_input("Tinggi Badan", min_value=0, max_value=200, value=160)
weight = st.number_input("Berat Badan", min_value=0, max_value=200, value=55)
bmi = st.number_input("BMI (Body Mass Index)", min_value=0, max_value=60, value=30)
physicalactivitylevel = st.number_input("Level Aktivitas", min_value=0, max_value=4, value=2)

if st.button("Hasil Diagnosa"):
    # Preprocessing input
    input_data = np.array([[age, height, weight, bmi, physicalactivitylevel]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_label = np.argmax(prediction)
    crop_name = label_encoder.inverse_transform([predicted_label])[0]

    st.success(f"Tes Obesitas: **{crop_name.upper()}**")

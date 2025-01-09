import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Memuat model, scaler, dan label encoder
model = pickle.load(open('Data.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

# Fungsi untuk melakukan prediksi
def predict(input_data):
    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)
    predicted_label = le.inverse_transform(prediction)
    return predicted_label[0]

# Judul aplikasi
st.title("Prediksi dengan Model Regresi Logistik")

# Input data oleh pengguna
st.write("Masukkan data untuk prediksi:")
input1 = st.number_input("Fitur 1", value=0.0)
input2 = st.number_input("Fitur 2", value=0.0)
input3 = st.number_input("Fitur 3", value=0.0)
input4 = st.number_input("Fitur 4", value=0.0)
input5 = st.number_input("Fitur 5", value=0.0)  # Sesuaikan dengan jumlah fitur yang ada pada dataset Anda

# Tombol untuk melakukan prediksi
if st.button('Prediksi'):
    # Mengambil data input pengguna
    input_data = [input1, input2, input3, input4, input5]  # Sesuaikan dengan jumlah fitur yang ada pada dataset Anda
    
    # Prediksi
    result = predict(input_data)
    
    # Menampilkan hasil
    st.write(f"Hasil Prediksi: {result}")

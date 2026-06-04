import streamlit as st
import numpy as np
import pandas as pd
import pickle

kmeans = pickle.load(open('kmeans.pkl','rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Customer Segmentation App")
st.write("Enter customer details to predict the segment")

age = st.number_input("Age", min_value = 18, max_value = 100, value = 35)
income = st.number_input("Income", min_value = 0, max_value = 200000, value = 50000)
total_spending = st.number_input("Total spending (sum of purchases)", min_value = 0, max_value = 5000, value = 1000)
num_web_purchases = st.number_input("Number of web purchases", min_value = 0, max_value = 100, value = 10)
num_store_purchases = st.number_input("Number of store purchases", min_value = 0, max_value = 100, value = 10)
num_web_visites = st.number_input("Number of web visits", min_value = 0, max_value = 50, value = 5)
recency = st.number_input("Recency (days since last purchase)", min_value = 0, max_value = 365, value = 30)

input = pd.DataFrame({
    'Age':[age],
    'Income':[income],
    'Total_spending':[total_spending],
    'NumWebPurchases':[num_web_purchases],
    'NumStorePurchases':[num_store_purchases],
    'NumWebVisitsMonth':[num_web_visites],
    'Recency':[recency]
})

input_scaled = scaler.transform(input)

if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    st.success(f"Predcited Cluster : Segment {cluster}")


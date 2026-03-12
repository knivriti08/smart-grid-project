import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

st.title("AI-Based Smart Grid Optimization")

# ---------------- Renewable Energy Forecasting ---------------- #

st.header("Renewable Energy Forecasting")

energy_type = st.selectbox(
    "Select Renewable Energy Source",
    ["Wind Energy", "Solar Energy", "Hydro Energy"]
)

wind = st.number_input("Enter Wind Speed")
temp = st.number_input("Enter Temperature")
sun = st.number_input("Enter Solar Irradiance")

data = pd.DataFrame({
    'WindSpeed':[5,7,10,12,6,8,15,9,11,14],
    'Temperature':[30,32,35,28,31,29,27,33,34,26],
    'Solar':[200,250,300,280,260,270,310,290,305,295],
    'PowerOutput':[100,150,300,350,120,200,400,220,310,380]
})

X = data[['WindSpeed','Temperature','Solar']]
y = data['PowerOutput']

model = LinearRegression()
model.fit(X,y)

if st.button("Predict Power Output"):
    prediction = model.predict([[wind,temp,sun]])
    st.success(f"Predicted Power Output: {prediction[0]:.2f}")

st.write("Selected Energy Source:", energy_type)

# ---------------- Energy Theft Detection ---------------- #

st.header("Energy Theft Detection")

consumption = st.number_input("Enter Consumption")

theft_data = pd.DataFrame({
    'Consumption':[100,120,500,130,600,110,105,700,115,125],
    'Theft':[0,0,1,0,1,0,0,1,0,0]
})

X = theft_data[['Consumption']]
y = theft_data['Theft']

clf = RandomForestClassifier(n_estimators=100,random_state=42)
clf.fit(X,y)

if st.button("Check Theft"):
    result = clf.predict([[consumption]])
    if result[0]==1:
        st.error("Theft Detected")
    else:
        st.success("Normal Consumption")

# ---------------- Fault Detection ---------------- #

st.header("Fault Detection")

voltage_data = np.array([[220],[222],[219],[400],[221],[405],[218],[410]])

kmeans = KMeans(n_clusters=2,n_init=10,random_state=42)
kmeans.fit(voltage_data)

if st.button("Show Cluster Labels"):
    st.write(kmeans.labels_)

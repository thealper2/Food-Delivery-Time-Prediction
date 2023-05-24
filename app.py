import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

model = tf.keras.models.load_model("FDTmodel.h5")

st.title("Food Delivery Time Prediction")

age = st.number_input("Age of Delivery Partner")
ratings = st.number_input("Ratings of Previous Deliveries")
distance = st.number_input("Total Distance")

if st.button("Predict"):
	test = np.array([[age, ratings, distance]])
	res = model.predict(test)
	print(res)
	st.success("Predicted Delivery Time in Minutes = " + str(res.item()))

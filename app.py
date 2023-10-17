
# Imports required ---
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from vega_datasets import data
from PIL import Image
import pickle5 as pickle
import sklearn

def main():

  st.title('Cardiovascular Disease Prediction')
  with st.expander("About this"):
       st.write("""
           This is KNN based predictor for cardiovascular disease. It has been trained on data obtained from Kaggle (https://www.kaggle.com/datasets/thedevastator/exploring-risk-factors-for-cardiovascular-diseas).""")
       st.write("""It takes into account 4 most relevant features that have high correlation with the presense (or absence) of cardiovascular diseases: Glucose, Cholesterol Level, Age, Weight. """)
  
  
  st.write("Please enter the following features: ")
  Age = st.number_input('Age', min_value=0, max_value=100, step=1, value=45)
  Weight = st.number_input('Weight (Kg)', value=75, min_value=30, max_value=250)
  level_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
  Glucose = st.radio('Glucose Level', [1,2,3])
  Cholestrol = st.radio('Cholestrol Level', [1,2,3])

  if st.button("Predict"):
    features = [Weight, Cholestrol, Glucose, Age]
    
    features_array = np.array(features)
    features_array = features_array.reshape(1, -1)
    
    scal = pickle.load(open("min_max_scaler.pkl", "rb"))
    model = pickle.load(open("heart_knn.pkl", "rb"))


    
    scaled_features = scal.transform(features_array)
    prediction = model.predict(scaled_features)[0]
    
    if prediction==1:
      st.write("Cardiovascular disease is likely to be present.")
    else:
      st.write("Congratulations! Cardivascular disease is likely to be absent.")


if __name__ == "__main__":
  main()
  
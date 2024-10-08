import streamlit as st
import pandas as pd
import joblib
import tensorflow.keras.saving
from sklearn.preprocessing import StandardScaler
import numpy

# Load the pre-trained model
model = tensorflow.keras.saving.load_model('titanic_model_s.keras')
scaler = StandardScaler()
# Create a Streamlit web application
st.title('Titanic Survival Prediction')

# Input form for passenger details
st.write('Enter Passenger Details:')
Pclass = st.selectbox('Pclass', [1, 2, 3])
Sex = st.selectbox('Sex', ['male', 'female'])
Age = st.slider('Age', 0, 100, 30)
Siblings_and_Spouses = st.number_input('SibSp', min_value=0, max_value=10, value=0)
Number_of_Parents_or_Children = st.number_input('Parch', min_value=0, max_value=10, value=0)
Fare = st.number_input('Fare', min_value=0.0, value=50.0)
#Embarked_Q = 1 if st.selectbox('Embarked_Q', ['Yes', 'No']) == 'Yes' else 0
#Embarked_S = 1 if st.selectbox('Embarked_S', ['Yes', 'No']) == 'Yes' else 0

# Make prediction based on input values

scaler.mean_ = [31.58649832, 32.20420797]
scaler.scale_ = [13.53633047, 49.66553444]
scaled_data = scaler.transform([[Age, Fare]])
input_data = [[Pclass, 1 if Sex == 'male' else 0, scaled_data[0][0], Siblings_and_Spouses, Number_of_Parents_or_Children, scaled_data[0][1]]]
#input_data = [[Pclass, 1 if Sex == 'male' else 0, Age, Siblings_and_Spouses, Number_of_Parents_or_Children, Fare]]
prediction = model.predict(numpy.array(input_data))

if st.button('Predict'):
    if prediction*100 >= 70:
        st.write('The passenger is likely to survive.')

    else:
        st.write('The passenger is unlikely to survive.')
    st.write(f'The passenger has a {prediction[0][0]*100:.2f}% chance of survival.')
    #st.write(f'{input_data}')
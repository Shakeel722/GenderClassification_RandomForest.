import pandas as pd
import numpy as np
import joblib
import sklearn

import streamlit as st

RF_model = r"C:\Users\Admin\AIML Jupyter folder\RandomForestModel.pkl"
my_model = joblib.load(RF_model) # here the directory of the model is passed indirectly by the variable 
# if you want to skip first step you can directly load passing the directory



# for title
st.header("SEX OF PENGUIN A RANDOM FOREST ML MODEL")


# culmen_length_mm	culmen_depth_mm	flipper_length_mm	body_mass_g	sex


culmen_length_mm = st.number_input("Please enter the culmen length (in mm) of the penguin")
culmen_depth_mm = st.number_input("Please enter the culmen depth  (in mm)  of the penguin ")
flipper_length_mm= st.number_input("Please enter the flipper lenght (in mm) of the penguin" )
body_mass_g = st.number_input("Please enter the  body mass(in grams) of the penguin" )


# applying MinMaxScaler as in real data to be intelligible for the model to work on it as we trained on Scaled data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_input =scaler.fit_transform ([[culmen_length_mm , 	culmen_depth_mm , flipper_length_mm  , body_mass_g	]])


x_data_scaled = np.array(scaled_input).astype("float64")


# making prediction and converting to original catagory
prediction = my_model.predict(x_data_scaled)


ans_dict = {0:"other"  , 1 : "female" , 2 : "male"}
final_ans = ans_dict[prediction[0]]

button = st.button("SUBMIT")

# Display the selected option

if(button):


   st.info("The model is predicting the sex of the penguin")
   st.write(f"As per the details provided we conclude that the sex of the penguin is : {final_ans}")
   




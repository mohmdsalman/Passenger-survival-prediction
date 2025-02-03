import pickle
import pandas as pd
import streamlit as st
    

st.title("Survival Classification")


Age = st.number_input("Age", min_value=0, max_value=100, value=20)
Gender = st.selectbox("Gender", ["Male", "Female"])
Class = st.selectbox("Class", ["Business", "Economy", "First"])
Seat_Type = st.selectbox("Seat Type", ["Aisle", "Middle", "Window"])
Fare_Paid = st.number_input("Fare Paid", min_value=0, max_value=1000, value=50)


input_data = pd.DataFrame([{
    "Age": Age,
    "Gender": Gender,
    "Class": Class, 
    "Seat_Type": Seat_Type,
    "Fare_Paid": Fare_Paid
}])

Gender_encoded = {"Female": 0, "Male": 1}
Class_encoded = {"Business": 0, "Economy": 1, "First": 2}
Seat_Type_encoded = {"Aisle": 0, "Middle": 1, "Window": 2}

input_data["Gender"] = input_data["Gender"].map(Gender_encoded)
input_data["Class"] = input_data["Class"].map(Class_encoded)
input_data["Seat_Type"] = input_data["Seat_Type"].map(Seat_Type_encoded)

df = pd.read_csv("features.csv")
feature_columns = [col for col in df.columns if col != 'Unnamed: 0']


input_data = input_data.reindex(columns=feature_columns, fill_value=0)

with open("model.pkl", "rb") as file:
    mdl = pickle.load(file)

prediction = mdl.predict(input_data)

if st.button('SUBMIT'):
    if prediction[0] == 1:
        st.success("Survived")
    else:
        st.error("Not Survived")


import streamlit as st 
import pickle
import numpy as np
import pandas as pd

def load_model():
    
    with open('diabetes_model_svm.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

classifier = data["model"]
sc = data["transform"]

def show_predict_page():
    
    st.title("Diabetes Checker")
    st.subheader("Enter details about the person")
    
    pregnancies = st.slider("Number of Pregnancies ",0,20,3)
    glucose = st.slider("Blood Glucose ",50,190,100)
    pressure = st.slider("Blood Pressure ",50,200,100)
    thickness = st.slider("Skin Thickness ",10,80,35) 
    ins = st.slider("Insulin ",10,250,35)      
    bmi = st.slider("B.M.I. ",15,50,25)
    pediFunc = st.text_input("Diabetes Pedigree Function ",0.350)
    age = st.slider("Age",1,90,35)              
    
    
    
    ok = st.button("Predict Diabetes Potential")
    if ok:
        x = np.array([[pregnancies,glucose,pressure,thickness,ins,bmi,pediFunc,age]])
        x = sc.transform(x)
        x = x.astype(float)
    
        Outcome = classifier.predict(x)
        st.subheader(f"The Probability of having Diabetes is {Outcome[0]}")
        
       
import streamlit as st
import numpy as np
import pickle

# define the stroke_prediction function
def stroke_prediction(input_data):
    # load the trained model
    loaded_model = pickle.load(open('trained_model.sav', 'rb'))
    # make a prediction
    prediction = loaded_model.predict([input_data])
    if prediction[0] == 0:
        return "This person is not likely to have a stroke."
    else:
        return "This person is likely to have a stroke."

def app():
    st.title(" Predicting Stroke likelihood :orange[System] :bar_chart:")
    st.write("You are in 'predict' page.")
    st.markdown("This Analysis dashboard lets us explore the relationship between **stroke** and the various "
                "variables in the dataset")

    # getting the input from the user
    sex = st.selectbox('Select Gender', ['Male', 'Female', 'Other'])
    age = st.slider('Age of the person', min_value=0, max_value=120, value=30)
    high_bp = st.selectbox('High blood pressure or Not?', ['Yes', 'No'])
    heart_disease = st.selectbox('Heart disease or Not?', ['Yes', 'No'])
    ever_married = st.selectbox('Ever been married?', ['Yes', 'No'])
    job_type = st.selectbox('Enter the job type', ['Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'])

    settlement = st.selectbox('Enter the settlement type', ['Urban', 'Rural'])
    avg_sugar_level = st.slider('Enter the average sugar/glucose level', min_value=0, max_value=500, value=100)
    body_mass_indx = st.slider('Enter the body mass index', min_value=0, max_value=100, value=25)
    smoking_status = st.selectbox('What is the smoking status?', ['Smokes', 'Never Smoked', 'Unknown'])

    # Encode the input values
    gender_dict = {'Male': 1, 'Female': 2, 'Other': 3}
    married_dict = {'Yes': 1, 'No': 0}
    work_dict = {'Private': 1, 'Self-employed': 2, 'Govt_job': 3, 'Children': 4, 'Never_worked': 5}
    residence_dict = {'Urban': 1, 'Rural': 0}
    smoking_dict = {'Smokes': 1, 'Never Smoked': 2, 'Unknown': 0}
    high_bp_dict = {'Yes': 1, 'No': 0}
    heart_disease_dict = {'Yes': 1, 'No': 0}

    gender = gender_dict[sex]
    married = married_dict[ever_married]
    work = work_dict[job_type]
    residence = residence_dict[settlement]
    smoking = smoking_dict[smoking_status]
    high_bp = high_bp_dict[high_bp]
    heart_disease = heart_disease_dict[heart_disease]

    # code for prediction
    diagnosis = ''
    # creating a button for prediction
    if st.button('Stroke likelihood'):
        input_data = (gender, age, high_bp, heart_disease, married, work, residence, avg_sugar_level,
                      body_mass_indx, smoking)
        diagnosis = stroke_prediction(input_data)
    st.success(diagnosis)





import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
import streamlit as st

filename = "claimAssit_cts.pkl"
loaded_model = pickle.load(open(filename, 'rb'))
jobtitle_le = pickle.load(open("le_job_title_Assist.pkl", 'rb'))
hereditary_disease_le = pickle.load(open("le_hereditary_diseases_Assist.pkl", 'rb'))
city_le = pickle.load(open("le_city_Assist.pkl", 'rb'))
scaler = pickle.load(open("scalerAssit.pkl", 'rb'))

def claim_prediction(input_data):
    # Convert input data into a DataFrame for processing
    input_df = pd.DataFrame([input_data], columns=['age', 'sex', 'weight', 'bmi', 'hereditary_diseases', 'no_of_dependents', 'smoker', 'city', 'bloodpressure', 'diabetes', 'regular_ex', 'job_title', 'claim'])

    # Replace categorical variables with numerical ones
    input_df['sex'].replace(['female', 'male'], [0, 1], inplace=True)
    input_df['smoker'].replace(['no', 'yes'], [0, 1], inplace=True)

    # Handling previously unseen labels with a special category or other appropriate action
    def safe_label_transform(le, column):
        known_labels = le.classes_
        # Replace unknown labels with a special category 'Unknown'
        input_df[column] = input_df[column].apply(lambda x: x if x in known_labels else 'Unknown')
        # Add 'Unknown' to label encoder classes if not present
        if 'Unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'Unknown')
        input_df[column] = le.transform(input_df[column])

    # Apply label encoding with handling for unseen labels
    safe_label_transform(hereditary_disease_le, 'hereditary_diseases')
    safe_label_transform(city_le, 'city')
    safe_label_transform(jobtitle_le, 'job_title')

    # Scale numerical features
    input_data_scaled = scaler.transform(input_df)

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(input_data_scaled)

    # Interpretation of the result
    if prediction[0] > 0.5 :
        return 'The claim is likely to be rejected.'
    else:
        return 'The claim is likely to be approved.'

def main():
    st.title('ClaimAssist Web Page')

    # Input fields
    age = st.text_input('Enter Age')
    sex = st.text_input('Enter Sex')
    weight = st.text_input('Enter Weight (in kg)')
    bmi = st.text_input('Enter Body Mass Index (BMI)')
    hereditary_diseases = st.text_input('List any hereditary diseases')
    no_of_dependents = st.text_input('No of Dependents')
    smoker = st.text_input('Do you smoke?(1:yes)(0:no)')
    city = st.text_input('Enter City')
    bloodpressure = st.text_input('Blood Pressure')
    diabetes = st.text_input('Do you have diabetes?(1:yes)(0:no)')
    regular_ex = st.text_input('Do you engage in regular exercise?(1:yes)(0:no)')
    job_title = st.text_input('Enter Job Title')
    claim = st.text_input('Enter Claim Amount')

    outcome = ''
    if st.button('Claim Prediction'):
        outcome = claim_prediction([age, sex, weight, bmi, hereditary_diseases, no_of_dependents, smoker, city, bloodpressure, diabetes, regular_ex, job_title, claim])
        st.success(outcome)

if __name__ == '__main__':
    main()


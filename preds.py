import streamlit as st
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

def main():
    st.title('Student Performance Predictor')

    # Form for user input
    st.subheader('Enter Student Details:')
    gender = st.selectbox('Gender', ['male', 'female'])
    race_ethnicity = st.selectbox('Race/Ethnicity', ['group A', 'group B', 'group C', 'group D', 'group E'])
    parental_level_of_education = st.selectbox('Parental Education Level', ['some high school', 'high school', 'some college', 'associate\'s degree', 'bachelor\'s degree', 'master\'s degree'])
    lunch = st.selectbox('Lunch Type', ['standard', 'free/reduced'])
    test_preparation_course = st.selectbox('Test Preparation Course', ['none', 'completed'])
    reading_score = st.slider('Reading Score', min_value=0, max_value=100, value=50)
    writing_score = st.slider('Writing Score', min_value=0, max_value=100, value=50)

    # Predict button
    if st.button('Predict'):
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )
        pred_df = data.get_data_as_data_frame()

        # Prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Show prediction result
        st.subheader('Prediction Result:')
        st.write(results[0])

if __name__ == '__main__':
    main()

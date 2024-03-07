
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import streamlit as st

# Load the trained model from joblib file
model = joblib.load('best_pipeline.joblib')

# Define feature names
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                  'BMI', 'DiabetesPedigreeFunction', 'Age']

# Create a title and an introduction for the app
st.title('Diabetes Prediction App')
st.markdown('This app predicts the likelihood of a patient having diabetes based on health and lifestyle factors.')

# Create a form for user inputs
with st.form('diabetes_prediction_form'):
    # Create two columns for input fields
    left_column, right_column = st.columns(2)

    # Add input fields to each column
    features_dict = {}
    for i, feature_name in enumerate(feature_names):
        if i % 2 == 0:
            value = left_column.number_input(feature_name, format='%f')
        else:
            value = right_column.number_input(feature_name, format='%f')
        features_dict[feature_name] = value

    # Add submit and cancel buttons
    submit_button = st.form_submit_button('Predict')
    cancel_button = st.form_submit_button('Cancel')

    # Process form submission
    if submit_button:
        # Collect input values into a NumPy array
        features = np.array([features_dict[feature_name] for feature_name in feature_names])
        features = features.reshape(1, -1)  # Reshape to a 2D array with one sample

        # Make prediction using the trained model
        prediction = model.predict(features)[0]

        # Display the prediction
        if prediction == 0:
            st.markdown('**Prediction:** No diabetes')
        else:
            st.markdown('**Prediction:** Diabetes')

    if cancel_button:
        st.session_state.clear()

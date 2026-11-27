import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# flask --app api.py run --port=5000
prediction_endpoint = "http://127.0.0.1:5000/predict"

st.title("Text Sentiment Predictor")

# Creates a file upload button in the UI.
uploaded_file = st.file_uploader("Choose a csv file for bulk prediction - Upload the file and click on Predict", type="csv")

# Text input for sentiment prediction
user_input = st.text_input("Enter text and click Predict", "")

# Prediction on single sentence
if st.button("Predict"):                                     # The block under it runs only when the button is clicked.
    if uploaded_file is not None:
        file = {"file": uploaded_file}
        response = requests.post(prediction_endpoint, files=file)
        response_bytes = BytesIO(response.content)
        response_df = pd.read_csv(response_bytes)

        st.download_button(label="Download Predcition", data=response_bytes, file_name="Prediction.csv", key="result_download_button")


    else:
        # response = requests.post(prediction_endpoint, data={'text': user_input})
        # response = response.json()
        # st.write(f"Predicted Sentiment: {response['prediction']}")

        
        response = requests.post(prediction_endpoint, json={"text": user_input})
        # quick sanity check before parsing JSON
        st.write("Status code from API:", response.status_code)
        st.write("Raw response (for debugging):", response.text)
        # only parse JSON if status is OK
        if response.status_code == 200:
            result = response.json()
            st.write(f"Predicted sentiment: {result.get('prediction')}")
        else:
            st.error("API returned an error. See raw response above.")


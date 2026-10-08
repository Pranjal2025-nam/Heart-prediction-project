# Heart-Disease Prediction

Predicts heart disease risk using Logistic Regression and a Streamlit app.

## Results
- Accuracy: ~87%
- Recall: ~94% (threshold = 0.3, to miss fewer patients)

## Steps
1. EDA and cleaning (Cholesterol zeros fixed)
2. One-hot encoding + StandardScaler
3. Logistic Regression with threshold tuning
4. Streamlit web app

## Run
pip install -r requirements.txt
streamlit run app.py

## Disclaimer
Student project. Not a medical diagnosis.

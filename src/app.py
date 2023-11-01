import streamlit as st
import pandas as pd
import numpy as np
import pickle

css_styles = '''
<style>
    body{
        font-family:"karla",
        "Helvetica Neue",
        sans-serif;
        font-size: 13px;
        }
    }

</style>
'''

st.set_page_config(page_title='Customer Prediction', 
                   layout='centered',page_icon="🧊")

st.markdown(css_styles,
            unsafe_allow_html=True)

with open('./src/model/xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to make predictions
def predict(transaction_amount, 
            account_balance, 
            gender, 
            R_score):
    
    data = pd.DataFrame({
        'account_balance': [account_balance],
        'transaction_amount': [transaction_amount],
        'gender_int': [gender],
        'R_score': [R_score]
    })

    data['gender_int'] = data['gender_int'].apply(lambda x: 1 if x == 'Female' else 0)

    prediction = model.predict(data)[0]
    return prediction


st.title('Customer Segmentation Prediction App')
st.write('Enter the details below to get the prediction.')
with st.form('Prediction_Form'):
    transaction_amount = st.number_input('Transaction Amount ($)', min_value=0.0)
    account_balance = st.number_input('Account Balance ($)', min_value=0.0)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    score = st.number_input('Recency Score', min_value=0, step=1)

    if st.form_submit_button('Predict Segment'):
        prediction = predict(transaction_amount, account_balance, gender, R_score=score)
        prediction_mapping = {0: 'A', 1: 'B', 2: 'C'}
        prediction = np.vectorize(prediction_mapping.get)(prediction, 'C')
        st.markdown(f'- Transaction Amount: {transaction_amount}')
        st.markdown(f'- Account Balance: {account_balance}')
        st.markdown(f'- Gender: {gender}')
        st.markdown(f'- Score: {score}')
        st.markdown(f'Customer Belongs to segment: **{prediction}**')



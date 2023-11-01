import pickle
import pandas as pd
import numpy as np


X_new_data = pd.DataFrame({
    'account_balance': [5300457], 
    'transaction_amount': [100], 
    'gender_int': [1], 
    'R_score': [5]
})

print(X_new_data)

with open('./src/model/xgboost_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)


predictions = loaded_model.predict(X_new_data)
prediction_mapping = {0: 'A', 1: 'B', 2: 'C'}
predictions = np.vectorize(prediction_mapping.get)(predictions, 'C')
print(predictions)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from config.config import hyperparameters, hyperparameters_test
plt.style.use("fivethirtyeight")


train = pd.read_csv('https://raw.githubusercontent.com/jblanco89/ML_Zoomcamp_Midterm/main/src/data/train_customer_segmented.csv', sep=',')
validation = pd.read_csv('https://raw.githubusercontent.com/jblanco89/ML_Zoomcamp_Midterm/main/src/data/validation_customer_segmented.csv', sep=',')
validation.head()

def tunning_hyper(train_df, validation_df):
    label_encoder = LabelEncoder()
    X_train = train_df[['account_balance', 'transaction_amount', 'gender_int']]
    X_val = validation_df[['account_balance', 'transaction_amount', 'gender_int']]
    validation_df['RF_segment'] = label_encoder.fit_transform(validation_df['RF_segment'])
    train_df['RF_segment'] = label_encoder.fit_transform(train_df['RF_segment'])
    y_val = validation_df['RF_segment']
    y_train = train_df['RF_segment']

    param_grid = hyperparameters
    # param_grid = hyperparameters_test
    
    smote = SMOTE(random_state=42,sampling_strategy='not minority')
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    xgb_model = xgb.XGBClassifier(objective='multi:softmax',
                                  num_classes=3)
    grid_search = GridSearchCV(estimator=xgb_model,
                               param_grid=param_grid,
                               cv=4, n_jobs=-1,
                               verbose=2)
    grid_search.fit(X_train_resampled, y_train_resampled)
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    report = classification_report(y_val, predictions, zero_division=1)
    y_prob = best_model.predict_proba(X_val)
    return accuracy, report, best_model

if __name__ == "__main__":
   accuracy, report, best_model = tunning_hyper(train_df=train,validation_df=validation)
   print("Accuracy:", accuracy)
   
   print("Report:")

   print(report)

   print("Best Model:", best_model)



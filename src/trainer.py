import pandas as pd
import xgboost as xgb
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

train_raw = pd.read_csv('./src/data/train_customer_segmented.csv', sep=',')
val_raw = pd.read_csv('./src/data/validation_customer_segmented.csv', sep=',')
test_raw = pd.read_csv('./src/data/test_customer_segmented.csv', sep=',')

full_train_df = pd.concat([train_raw, val_raw], axis=0)

def train_xgboost_model(train_df, test_df):
    label_encoder = LabelEncoder()
    X_train = train_df[['account_balance', 'transaction_amount', 'gender_int', 'R_score']]
    X_test = test_df[['account_balance', 'transaction_amount', 'gender_int', 'R_score']]
    test_df['RF_segment'] = label_encoder.fit_transform(test_df['RF_segment'])
    train_df['RF_segment'] = label_encoder.fit_transform(train_df['RF_segment'])
    y_test = test_df[['RF_segment']]
    y_train = train_df[['RF_segment']]

    smote = SMOTE(random_state=42,sampling_strategy='not minority')
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model = xgb.XGBClassifier(max_depth=10,
                              learning_rate=0.1,
                              n_stimators = 400,
                              objective='multi:softmax',
                              subsample = 0.7,
                              num_classes=3)
    
    model.fit(X_train_resampled, y_train_resampled)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    with open('./src/model/xgboost_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    return accuracy, report


if __name__ == "__main__":
    accuracy, report = train_xgboost_model(train_df=full_train_df, 
                                           test_df=test_raw)
    print("Accuracy:", accuracy)
   
    print("Report:")

    print(report)



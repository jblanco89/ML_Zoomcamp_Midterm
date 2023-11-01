column_dict_to_rename = {'TransactionAmount (INR)':'transaction_amount',
                        'Age': 'age',
                        'GenderNumeric': 'gender_int',
                        'TransactionDate': 'transaction_date',
                        'CustLocation': 'location',
                        'CustomerID' : 'customer_id',
                        'TransactionID': 'transaction_id',
                        'CustAccountBalance': 'account_balance',
                        'TransactionTime':'transaction_time'}

feature_list = ['account_balance', 'transaction_amount', 'gender_int', 'R_score']

target_var = ['RF_segment']

hyperparameters = {
            'n_estimators': [100, 300, 400],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.1, 1, 0.01],
            # 'colsample_bytree': [0.8, 0.9, 1.0],
            'subsample': [0.7, 0.9],
        }

hyperparameters_test = {
            'n_estimators': [300],
            'max_depth': [4],
            'learning_rate': [0.001],
            'colsample_bytree': [1.0],
            'subsample': [0.8],
        }
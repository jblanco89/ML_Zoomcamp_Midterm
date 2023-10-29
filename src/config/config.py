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
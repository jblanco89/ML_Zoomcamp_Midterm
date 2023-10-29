import numpy as np
import pandas as pd
from config.config import column_dict_to_rename
from config.config import feature_list
from config.config import target_var


def filter_and_rename(df, column_dict):
    df = df.rename(columns=column_dict)
    df = df[df['age'] < 99]
    return df

def remove_nans(df):
    df = df.dropna()
    return df

def transform_data(df):
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%d/%m/%Y')
    return df

def rf_metrics(df, metrics:list = ['recency', 'frequency']):
    df_temp = df.drop(columns='transaction_date')
    df_temp = df_temp.groupby('customer_id').agg({
    'transaction_id':'count',
    'age':'first',
    'gender_int':'first',
    'location':'first',
    'account_balance':'mean',
    'transaction_amount':'mean'
    })
    for i in metrics:
        if i == 'recency':
            max_transaction_date = df.groupby('customer_id').transaction_date.max().reset_index()
            max_transaction_date.columns = ['customer_id','max_transaction_date']
            df_temp = df_temp.reset_index()
            max_transaction_date['recency'] = (max_transaction_date['max_transaction_date'].max() - 
                                   max_transaction_date['max_transaction_date']).dt.days
            df = pd.merge(df, max_transaction_date[['customer_id', 'recency']], on='customer_id')
        if i == 'frequency':
            user_frequency = df.groupby('customer_id').transaction_date.count().reset_index()
            user_frequency.columns = ['customer_id','frequency']
            df_temp = df_temp.reset_index()
            df = pd.merge(df, user_frequency, on='customer_id')

    # df = pd.concat([df])        
    return df

def rf_scores(df, r_weight, f_weight, q):
    if np.sum([r_weight, f_weight]) == 1.0:
        weight_R = r_weight
        weight_F = f_weight
        df['R_score'] = pd.qcut(df['recency'], 
                                duplicates='drop', 
                                q=q, labels=False)
        df['F_score'] = pd.qcut(df['frequency'], q=q,
                                duplicates='drop',
                                labels=False)
        df['RF_score'] = (df['R_score'] * weight_R +
                            df['F_score'] * weight_F)
    else:
        print('Weight ratio sum must be equal to 1')

    return df

def rf_segments(df):
    segments = {
        'A': (df['RF_score'] >= df['RF_score'].quantile(0.75)),
        'B': (df['RF_score'] >= df['RF_score'].quantile(0.25)) &
     (df['RF_score'] < df['RF_score'].quantile(0.75)),
        'C': (df['RF_score'] < df['RF_score'].quantile(0.25))
    }
    df['RF_segment'] = np.select(segments.values(), segments.keys(), default='Other')

    return df

def select_features(df, feature_list:list, target:list):
    df = df[feature_list + target]
    return df

def generate_data_output(df,
                         prefix:str = 'train', 
                         name:str = 'output_data',
                         out_path:str = 'src/data',
                         format_file:str = 'csv', 
                         index:bool = False,
                         header:bool=True):
    df = df.reset_index()
    if format_file == 'csv':
        df.to_csv(out_path + '/' + prefix + '_'+ name + '.' + format_file, 
                  sep=',', 
                  header=header,
                  index=index)
    if format_file == 'xlsx':
        df.to_excel(out_path + '/' + prefix + '_'+ name + '.' + format_file, 
                  sheet_name = 'output_file', 
                  header=header,
                  index=index)
        
    return print('new file has been generated')


def main():
    column_dict = column_dict_to_rename
    features = feature_list
    target = target_var
    file_list = ['train.csv', 'validation.csv', 'test.csv']
    # file_list = ['train.csv']
    for f in file_list:
        data_raw = pd.read_csv(f'./dataset/{f}', sep=';')
        df = filter_and_rename(df=data_raw, column_dict=column_dict)
        df = remove_nans(df=df)
        df = transform_data(df=df)
        df = rf_metrics(df=df)
        df = rf_scores(df=df, r_weight=0.3, f_weight=0.7, q=10)
        df = rf_segments(df=df)
        df = select_features(df=df, feature_list=features, target=target)
        df = generate_data_output(df=df, prefix=f[:-4],name='customer_segmented')
    # print(df[['recency', 'frequency']].describe())
    # print(df.head())
    # return df

if __name__ == "__main__":
    main()







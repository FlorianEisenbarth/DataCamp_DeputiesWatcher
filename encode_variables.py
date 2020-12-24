import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder



def numerical_encoding(df, max_categories=20):
    ''' Add numerical equivalent columns to the dataframe df for every categorical 
        features in df with a number of categories less than max_categories.

        Return is the extended transformed dataframe
    '''
    transformed_df = df.copy()
    ord_enc = OrdinalEncoder()
    columns = df.columns.copy()
    for col in columns:
        if df[col].dtype != 'float64' and len(set(df[col])) <= max_categories:
        # data is considered categorical kind
            transformed_df[col + '_numerical'] = ord_enc.fit_transform(df[[col]])

    return transformed_df

def one_hot_encoding(categorical_list):
    ''' Turns a list-like object (such as pandas.Serie) of categorical entries into
        its one-hot-encoded equivalent.

        Return is a 2D array of one-hot-encoded vectors
    '''
    oh_enc = OneHotEncoder(sparse=False)
    transformed_list = oh_enc.fit_transform(np.array(categorical_list).reshape(-1,1))

    return transformed_list

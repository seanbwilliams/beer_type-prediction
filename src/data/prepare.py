# A collection of helpful data preparation functions

import pandas as pd
import numpy as np

def select_features(keep_cols, df_train, df_test=None):
    """Keep selected cols in dataframes

    Parameters
    ----------
    keep_cols : List
        List of columns to keep
    df_train : pd.DataFrame
        Training Dataframe containing all features
    df_test : pd.DataFrame
        Test Dataframe containing all features

    Returns
    -------
    Pandas Dataframe
        Dataframe containing train dataset
    Pandas Dataframe
        Dataframe containing test dataset
    """

    df_train_copy = df_train[keep_cols].copy()
    if df_test is not None:
        df_test_copy = df_test[keep_cols].copy()
    else:
        df_test_copy = None

    return df_train_copy, df_test_copy

def ordinal_encoding(cats_dict, df_train, df_test=None): 
    """Apply ordinal encoding to do supplied list of column

    Parameters
    ----------
    cats_dict : Dictionary
        Dictionay of columns to encode with ordinal map
    df_train : pd.DataFrame
        Training Dataframe to apply encoding
    df_test : pd.DataFrame
        Test Dataframe to apply encoding

    Returns
    -------
    Pandas Dataframe
        Dataframe containing encoded train data
    Pandas Dataframe
        Dataframe containing encoded test data
    """
    from sklearn.preprocessing import OrdinalEncoder

    for col, cats in cats_dict.items():
        encoder = OrdinalEncoder(categories=cats)
        df_train[col] = encoder.fit_transform(df_train[[col]])
        if df_test is not None:
            df_test[col] = encoder.transform(df_test[[col]])
        else:
            df_test = None

    return df_train, df_test

def convert_to_date(df, cols:list):
    """Convert specified columns from a Pandas dataframe into datetime

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    cols : list
        List of columns to be converted

    Returns
    -------
    pd.DataFrame
        Pandas dataframe with converted columns
    """
    
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit='s')
    return df

def impute_missing_values(impute_strategy, df_train, df_test=None):
    if impute_strategy == 'mode':
        fillval = df_train['beer_abv'].mode().max()
        df_train['beer_abv'] = df_train['beer_abv'].fillna(fillval)

        if df_test is not None:
            fillval = df_test['beer_abv'].mode().max()
            df_test['beer_abv'] = df_test['beer_abv'].fillna(fillval)
        else:
            df_test = None

    #from sklearn.impute import SimpleImputer   
    #imputer = SimpleImputer(missing_values=np.nan, strategy=impute_strategy)
    #df_train = imputer.fit_transform(df_train)
    #if df_test is not None:
    #    df_test = imputer.transform(df_test)
    #else:
    #    df_test = None
    
    return df_train, df_test

def scale_features(df_in, scaler, target_col=None):
    """Split sets randomly

    Parameters
    ----------
    df_in : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    scaler : float
        Scaling processor to fit to features

    Returns
    -------
    X : pd.DataFrame
        Dataframe containing scaled data
    """

    X = df_in.copy()
    if target_col is not None:
        y = X.pop(target_col)

    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(np.squeeze(X_scaled), columns=X.columns)
    if target_col is not None:
        X = pd.concat([X_scaled, y], axis=1)
    else:
        X = X_scaled

    return X
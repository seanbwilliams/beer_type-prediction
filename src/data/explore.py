# A collection of helpful data exploration functions

import pandas as pd
import numpy as np

def read_csv_data(train_filename, test_filename=None, path='../data/raw/'):
    """Load CSV data into dataframe

    Parameters
    ----------
    train_filename : str
        Name of the file containing train dataset
    test_filename : str
        Name of the file containing test dataset
    path : str
        Path to the folder where the raw files are saved (default: '../data/processed/')

    Returns
    -------
    Pandas Dataframe
        Dataframe containing train dataset
    Pandas Dataframe
        Dataframe containing test dataset
    """

    if train_filename is not None:
      df_train = pd.read_csv(path + train_filename)
    else:
      df_train = None
    if test_filename is not None:
      df_test = pd.read_csv(path + test_filename)
    else:
      df_test = None

    return df_train, df_test

def print_info(df_in, rows_to_display=20):
    """Print summary information, shape, head and tail of a dataframe

    Parameters
    ----------
    df_in : pd.DataFrame
        Dataframe 

    Returns
    -------
    info,  shape,  head and tail of the dataset
    """
    print('**********************')
    print('Dataframe Information:')
    print('**********************')
    print(df_in.info())
    print('')

    print('****************')
    print('Dataframe Shape:')
    print('****************')
    print('Dataframe (rows, columns): ', df_in.shape)
    print('')

    print('***************')
    print('Dataframe Data:')
    print('***************')
    display(df_in.head(rows_to_display), df_in.tail(rows_to_display))

    print('****************')
    print('Dataframe Stats:')
    print('****************')
    display(df_in.describe())

def print_na_info(df_in):
    """Print information about null values in data set

    Parameters
    ----------
    df_in : pd.DataFrame
        Dataframe containing features to be interrogated

    Returns
    -------
    N/A
    """

    isna_sum_sorted = df_in.isna().sum().sort_values(ascending=False)
    len_df_in = len(df_in)
    df_feat_missing = pd.DataFrame ({
                      'feature': isna_sum_sorted.index,
                      'missing value count': isna_sum_sorted.values,
                      '% of total': 100 * isna_sum_sorted.values / len_df_in
                      })
    df_feat_missing.drop(df_feat_missing[df_feat_missing['missing value count']== 0].index, inplace=True)
    df_feat_missing.reset_index(inplace=True, drop=True)
    display(df_feat_missing)

def print_negative_info(df_in):
    """Print information about negative values in data set

    Parameters
    ----------
    df_in : pd.DataFrame
        Dataframe containing features to be interrogated

    Returns
    -------
    N/A
    """

    neg_sum_sorted = (df_in < 0).sum().sort_values(ascending=False)
    len_df_in = len(df_in)
    df_feat_neg = pd.DataFrame ({
                  'feature': neg_sum_sorted.index,
                  'negative value count': neg_sum_sorted.values,
                  '% of total': 100 * neg_sum_sorted.values / len_df_in
                  })
    df_feat_neg.drop(df_feat_neg[df_feat_neg['negative value count']== 0].index, inplace=True)
    df_feat_neg.reset_index(inplace=True, drop=True)
    display(df_feat_neg)


def print_duplicate_info(df_in, uid_col=None):
    """Print information about duplicate observations in data set

    Parameters
    ----------
    df_in : pd.DataFrame
        Dataframe containing features to be interrogated

    Returns
    -------
    N/A
    """

    if uid_col is not None:
        print('There are %d of duplicated columns.' % (len(df_in[uid_col]) - len(df_in[uid_col].unique())))

    df = df_in[df_in.duplicated(keep=False)]
    display(df)

def print_unique_info(df_in, col_list):
    """Print unqiue values for each column in data set

    Parameters
    ----------
    df_in : pd.DataFrame
        Dataframe containing features to be interrogated
    col_list : list
        List of columns which to interrogate

    Returns
    -------
    N/A
    """

    unique_values={}
    for col in col_list:
        unique_values[col] = df_in[col].unique()
    df_unique_values = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in unique_values.items() ]))
    display(df_unique_values.fillna(''))
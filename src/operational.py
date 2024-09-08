from config import config

import pandas as pd
import numpy as np

def negative_to_zero(values):
    """
    Function to post process forecasted generation to not enable negative values
    """
    return np.where(values < 0, 0, values)

def process_features_dataframe(df):
    """
    Receive a feature dataframe, build the aditional and return for a new forecast
    """
    cfg = config['predict']
    df.index = pd.to_datetime(df.index)
    df = build_features(df)
    df = df.T.reindex(index=cfg['columns_order']).T
    return df

def load_features_dataframe():
    """
    Read a sample file with the features, build the aditional and return for a new forecast
    """
    df = pd.read_csv('data/ec_features_2024-06-15.csv', sep=';', index_col=0)
    df = process_features_dataframe(df)
    return df

def load_train_dataframe():
    """
    Read the file with the features and the file with the target, could be a consume from a database
    """
    features = pd.read_csv('data/ec_forecast_ufv_bom_jesus.csv', sep=';', index_col=0)
    target = pd.read_csv('data/ufv_bom_jesus_da_lapa_gen.csv', sep=',',index_col=0)

    df = features.merge(target, right_index=True, left_index=True)
    df.index = pd.to_datetime(df.index)

    df = build_features(df)
    return df

def build_features(df):
    """
    Build aditional features
    """
    df = df.copy()
    df['sin(hour)'] = np.sin(df.index.hour / 24 * 2 * np.pi)
    df['cos(hour)'] = np.cos(df.index.hour / 24 * 2 * np.pi)
    return df


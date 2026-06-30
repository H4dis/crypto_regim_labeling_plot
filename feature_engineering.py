import pandas as pd
import numpy as np

def create_advanced_features(df):
    df = df.copy()

    df['return_1'] = df['SS_price_hourly'].pct_change()
    df['return_3'] = df['SS_price_hourly'].pct_change(periods=3)
    df['return_5'] = df['SS_price_hourly'].pct_change(periods=5)

    df['volume_ma_10'] = df['SS_volume_hourly'].rolling(10).mean()
    df['volume_ratio'] = df['SS_volume_hourly'] / df['volume_ma_10']

    df['volatility_5'] = df['SS_price_hourly'].rolling(5).std()
    df['volatility_10'] = df['SS_price_hourly'].rolling(10).std()

    delta = df['SS_price_hourly'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    df['ma_20'] = df['SS_price_hourly'].rolling(20).mean()
    df['distance_from_ma'] = (df['SS_price_hourly'] - df['ma_20']) / df['ma_20'] * 100

    df['slope_5'] = df['SS_price_hourly'].diff(5) / 5

    df['new_high_5'] = (df['SS_price_hourly'] == df['SS_price_hourly'].rolling(5).max()).astype(int)
    df['new_low_5'] = (df['SS_price_hourly'] == df['SS_price_hourly'].rolling(5).min()).astype(int)

    return df
FEATURES = ['return_1', 'return_3', 'return_5', 'volume_ratio',
            'volatility_5', 'volatility_10', 'rsi_14', 'distance_from_ma',
            'slope_5', 'new_high_5', 'new_low_5']

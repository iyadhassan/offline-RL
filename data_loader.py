import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def fetch_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    return data.reset_index()

def preprocess_data(data, train_ratio=0.8):
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['Returns'] = data['Close'].pct_change()
    data = data.dropna()
    
    scaler = StandardScaler()
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'Returns']
    data.loc[:, features] = scaler.fit_transform(data[features])
    
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size].copy()
    test_data = data[train_size:].copy()
    
    return train_data, test_data, scaler

def load_data(symbol, start_date, end_date, train_ratio=0.8):
    data = fetch_stock_data(symbol, start_date, end_date)
    train_data, test_data, scaler = preprocess_data(data, train_ratio)
    return train_data, test_data, scaler
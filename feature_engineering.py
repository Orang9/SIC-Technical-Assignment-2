import pandas as pd
from sklearn.preprocessing import StandardScaler

def explore_data(data):
    print("Columns:", data.columns)
    print(data.info())
    print(data.describe())
    print(data.head())

import pandas as pd
from sklearn.preprocessing import StandardScaler

def encode_features(data):
    data = pd.get_dummies(data, columns=['Type', 'Product ID'])
    return data

def scale_features(data):
    scaler = StandardScaler()
    columns_to_drop = ['UDI']  # Adjust columns to drop as needed
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    scaled_features = scaler.fit_transform(data.drop(columns=columns_to_drop, axis=1))
    return pd.DataFrame(scaled_features, columns=data.columns.drop(columns_to_drop))

def prepare_features(data):
    data = encode_features(data)
    data = scale_features(data)
    return data
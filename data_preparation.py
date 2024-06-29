import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def explore_data(data):
    print(data.info())
    print(data.describe())
    print(data.head())

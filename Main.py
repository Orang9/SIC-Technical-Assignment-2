from data_preparation import load_data, explore_data
from feature_engineering import prepare_features
from model_training import train_model
from model_tuning import tune_model
from model_deployment import save_model, load_model, create_app
import pandas as pd

file_path = 'ai4i2020.csv'

data = load_data(file_path)
explore_data(data)

sampled_data = data.sample(n=10, random_state=20)

processed_data = prepare_features(sampled_data)

if 'TWF' in processed_data.columns:
    X = processed_data.drop(columns=['TWF'])
    y = pd.to_numeric(processed_data['TWF'], downcast='integer', errors='coerce')
    y = y.dropna().astype(int)
    
    trained_model = train_model(X, y)
else:
    print("Column 'TWF' not found in processed data.")
    exit()  

tuned_model = tune_model(trained_model, X, y)

save_model(tuned_model, 'best_rf_model.pkl')

model = load_model('best_rf_model.pkl')
app = create_app(model)
app.run(port=5000, debug=True)

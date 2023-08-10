import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv(r"/Users/sudeepdharanikota/Desktop/HouseDataProj/processeddata.csv")

# Split features and target
X = df.drop(columns=['price'])
y = df['price']

# One hot encode zip_code
encoder = OneHotEncoder()
encoded_zip = encoder.fit_transform(X['zip_code'].values.reshape(-1, 1)).toarray()
encoded_zip_df = pd.DataFrame(encoded_zip, columns=[f'zip_{cat}' for cat in encoder.categories_[0]])
X = pd.concat([X.drop(columns='zip_code'), encoded_zip_df], axis=1)

# Normalize other features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the hyperparameters and their possible values for tuning
param_dist = {
    'n_estimators': [100, 200, 500, 1000],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}


# Initialize the RandomForestRegressor
rf_regressor = RandomForestRegressor(random_state=42, n_jobs=-1)

# Use RandomizedSearchCV to search for best hyperparameters
rf_search = RandomizedSearchCV(estimator=rf_regressor, param_distributions=param_dist, 
                               n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Fit the random search model
rf_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters from RandomizedSearchCV: ", rf_search.best_params_)

# Use the best estimator from RandomizedSearchCV to make predictions
best_rf = rf_search.best_estimator_
y_pred_rf = best_rf.predict(X_val)

# Calculate evaluation metrics
mae_rf = mean_absolute_error(y_val, y_pred_rf)
mse_rf = mean_squared_error(y_val, y_pred_rf)
r2_rf = r2_score(y_val, y_pred_rf)

print(f"Random Forest - Mean Absolute Error: {mae_rf}")
print(f"Random Forest - Mean Squared Error: {mse_rf}")
print(f"Random Forest - R-squared: {r2_rf}")

# Make a prediction using the best RandomForestRegressor
sample_input_rf = X_val[0].reshape(1, -1)
predicted_price_rf = best_rf.predict(sample_input_rf)
print(f"Predicted Price using Random Forest: {predicted_price_rf[0]}")

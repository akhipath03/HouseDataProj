import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

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

# Initialize the model
model = Sequential()

# Input Layer
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

# Hidden Layers
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

# Output Layer
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=10)

# Predict on the validation set
y_pred = model.predict(X_val)

# Calculate r^2
r2 = r2_score(y_val, y_pred)
print(f"R-squared: {r2}")

# Make a prediction
sample_input = np.array([X_val[0]])  # Replace with your input data
predicted_price = model.predict(sample_input)
print(f"Predicted Price: {predicted_price[0][0]}")

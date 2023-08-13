import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout

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

# Define the model architecture
model = Sequential()

# Input Layer
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))

# Hidden Layers
model.add(Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Dropout(0.2))

model.add(Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(1, activation='linear'))

# Compile the model with Adam optimizer
model.compile(optimizer=keras.optimizers. Adam(learning_rate=1e-3), loss='mean_squared_error')

# Additional Callbacks
early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), 
          epochs=100, batch_size=32, 
          callbacks=[early_stopping, reduce_lr])

# Predict on the validation set
y_pred = model.predict(X_val)

# Calculate evaluation metrics
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"Modified Model - Mean Absolute Error: {mae}")
print(f"Modified Model - Mean Squared Error: {mse}")
print(f"Modified Model - R-squared: {r2}")

# Make a prediction
sample_input = np.array([X_val[0]])  # Replace with your input data
predicted_price = model.predict(sample_input)
print(f"Predicted Price: {predicted_price[0][0]}")

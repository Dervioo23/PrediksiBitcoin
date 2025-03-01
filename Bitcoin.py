import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
# Bagian 1: Pembelajaran dan Evaluasi Model
# ------------------------------------------
# Load historical data
file_path = "bitcoin_gabungan.xlsx"
df = pd.read_excel(file_path)
df
# Menghitung dan menghapus duplikat berdasarkan kolom 'timestamp'
duplicates_by_column = df.duplicated(subset=["timestamp"]).sum()
print(f"Jumlah duplikat berdasarkan kolom 'timestamp': {duplicates_by_column}")
# Menghapus data duplikat berdasarkan kolom 'timestamp'
df_clean = df.drop_duplicates(subset=["timestamp"])
# Mengecek ulang jumlah duplikat setelah penghapusan
data_clean = df_clean.duplicated(subset=["timestamp"]).sum()
print(f"Jumlah duplikat setelah penghapusan: {data_clean}")
# Pastikan kolom 'timestamp' diubah menjadi tipe datetime
df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
df_clean
# Normalize the 'close' price
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df_clean['close'].values.reshape(-1, 1))
# Create sequences
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

seq_length = 50
X, y = create_sequences(data_scaled, seq_length)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
# Split data into training and validation sets
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]
print(f"Ukuran X_train: {X_train.shape}")
print(f"Ukuran y_train: {y_train.shape}")
print(f"Ukuran X_val: {X_val.shape}")
print(f"Ukuran y_val: {y_val.shape}")
# Build CNN-LSTM model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, 1)),
    LSTM(units=50, return_sequences=True),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
# Save best model using ModelCheckpoint
checkpoint = ModelCheckpoint("bitcoin_model.keras", save_best_only=True, monitor='val_loss', mode='min')
# Train model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=16, callbacks=[checkpoint])
# Evaluate model on validation data
model = load_model("bitcoin_model.keras")
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print(f"Validation MSE: {mse}")
print(f"Validation R²: {r2}")
# Create results DataFrame
results_cnn_lstm = pd.DataFrame({'Actual': y_val.flatten(), 'Predicted_CNN_LSTM': y_pred.flatten()})
print("\nHasil Prediksi CNN-LSTM:")
print(results_cnn_lstm.head())
# Visualize predictions
plt.figure(figsize=(12, 6))
plt.plot(y_val, label="Actual Prices", color="blue")
plt.plot(y_pred, label="Predicted Prices", color="red", linestyle="dashed")
plt.xlabel("Time")
plt.ylabel("Normalized Price")
plt.legend()
plt.title("Validation: Actual vs Predicted")
plt.show()
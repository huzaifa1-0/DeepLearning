import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ================= CONFIG =================
TICKER = "AAPL"
START_DATE = "2018-01-01"
END_DATE = "2023-12-31"
SEQ_LEN = 60
EPOCHS = 30
BATCH_SIZE = 32

# ================= DATA =================
def fetch_data():
    df = yf.download(TICKER, start=START_DATE, end=END_DATE)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df.dropna()

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 3])  # Close price
    return np.array(X), np.array(y)

# ================= PIPELINE =================
df = fetch_data()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

X, y = create_sequences(scaled_data, SEQ_LEN)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ================= MODEL =================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# ================= TRAIN =================
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# ================= EVALUATE =================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# ================= PLOT =================
plt.figure(figsize=(10,5))
plt.plot(y_test[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted")
plt.legend()
plt.title(f"{TICKER} Price Prediction")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def create_sine_wave_data(seq_length=1000):
    """Create synthetic time series data"""
    x = np.linspace(0, 50, seq_length)
    y = np.sin(x) + np.random.normal(0, 0.1, seq_length)  # sine wave with noise
    
    # Plot the data
    plt.figure(figsize=(12, 4))
    plt.plot(y[:200], label='Sine Wave with Noise')
    plt.title('Synthetic Time Series Data (First 200 points)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    
    return y

# Generate data
time_series = create_sine_wave_data()

# ==================== STEP 2: PREPARE THE DATA ====================
def prepare_data(data, look_back=20):
    """
    Prepare data for RNN
    look_back: How many previous time steps to use for prediction
    """
    # Normalize the data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_normalized) - look_back):
        X.append(data_normalized[i:(i + look_back), 0])
        y.append(data_normalized[i + look_back, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for RNN: [samples, time_steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Prepare data
look_back = 20  # Use last 20 points to predict next point
X, y, scaler = prepare_data(time_series, look_back)

# Split into train and test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Each sequence has {look_back} time steps")

# ==================== STEP 3: BUILD THE RNN MODEL ====================
def build_rnn_model(look_back=20):
    """Build a simple RNN model"""
    model = Sequential([
        # First RNN layer
        SimpleRNN(50, activation='tanh', return_sequences=True, 
                 input_shape=(look_back, 1)),
        Dropout(0.2),  # Prevent overfitting
        
        # Second RNN layer
        SimpleRNN(50, activation='tanh', return_sequences=False),
        Dropout(0.2),
        
        # Output layer (predict next value)
        Dense(1)
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mse',  # Mean Squared Error for regression
        metrics=['mae']  # Mean Absolute Error
    )
    
    return model

# Build model
model = build_rnn_model(look_back)
model.summary()

# ==================== STEP 4: TRAIN THE MODEL ====================
print("\nTraining the RNN model...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# ==================== STEP 5: MAKE PREDICTIONS ====================
# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform to original scale
train_predict = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# ==================== STEP 6: VISUALIZE RESULTS ====================
plt.figure(figsize=(15, 6))

# Create time axis for plotting
train_time = np.arange(look_back, look_back + len(train_predict))
test_time = np.arange(len(train_predict) + 2*look_back, 
                      len(train_predict) + 2*look_back + len(test_predict))

# Plot training predictions
plt.subplot(1, 2, 1)
plt.plot(train_time, y_train_inv, label='Actual (Train)', alpha=0.7)
plt.plot(train_time, train_predict, label='Predicted (Train)', alpha=0.7)
plt.title('Training Data: Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

# Plot testing predictions
plt.subplot(1, 2, 2)
plt.plot(test_time, y_test_inv, label='Actual (Test)', alpha=0.7)
plt.plot(test_time, test_predict, label='Predicted (Test)', alpha=0.7)
plt.title('Testing Data: Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()

# ==================== STEP 7: EVALUATE THE MODEL ====================
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate metrics
train_mae = mean_absolute_error(y_train_inv, train_predict)
test_mae = mean_absolute_error(y_test_inv, test_predict)
train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict))

print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)
print(f"Training MAE:  {train_mae:.4f}")
print(f"Testing MAE:   {test_mae:.4f}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing RMSE:  {test_rmse:.4f}")

# ==================== STEP 8: MAKE FUTURE PREDICTIONS ====================
# Predict next 50 steps
def predict_future(model, last_sequence, n_future=50):
    """
    Predict future values using the model
    """
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_future):
        # Reshape for prediction
        current_input = current_sequence.reshape((1, look_back, 1))
        
        # Predict next value
        next_pred = model.predict(current_input, verbose=0)
        
        # Store prediction
        future_predictions.append(next_pred[0, 0])
        
        # Update sequence (remove first, add prediction)
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred[0, 0]
    
    return np.array(future_predictions)

# Get last sequence from test data
last_sequence = X_test[-1].flatten()

# Predict future
n_future = 50
future_predictions = predict_future(model, last_sequence, n_future)

# Inverse transform
future_predictions = scaler.inverse_transform(future_predictions.reshape(-1, 1))

# Plot future predictions
plt.figure(figsize=(12, 5))
future_time = np.arange(len(time_series), len(time_series) + n_future)

plt.plot(time_series[-100:], label='Last 100 Actual Values', alpha=0.7)
plt.plot(future_time, future_predictions, 'r--', label='Future Predictions', alpha=0.7)
plt.title('Future Predictions (Next 50 Steps)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

print("\n" + "="*50)
print("BEGINNER EXPLANATIONS:")
print("="*50)
print("1. RNNs are good for sequence data (time series, text, etc.)")
print("2. We use 'look_back' to decide how much history to consider")
print("3. SimpleRNN is the basic RNN layer (better options: LSTM, GRU)")
print("4. We normalize data to help the model learn better")
print("5. The model learns patterns to predict the next value")
print("6. Dropout layers prevent overfitting (memorizing training data)")
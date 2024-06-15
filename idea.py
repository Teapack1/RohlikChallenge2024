import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from kerastuner.tuners import RandomSearch

# Load and preprocess data (assuming train_df is already loaded and processed)
# Ensure you have the 'orders' column in the DataFrame

# Define features
features = ['orders_lag_1', 'orders_lag_7', 'orders_lag_30', 
            'orders_roll_mean_7', 'orders_roll_std_7', 
            'orders_roll_mean_30', 'orders_roll_std_30',
            'year', 'month', 'day', 'day_of_week', 'is_weekend']
train_data_lstm = train_df[features]

# Function to prepare data for LSTM
def prepare_data_for_lstm(df, target_col_index, n_timesteps):
    X, y = [], []
    for i in range(n_timesteps, len(df)):
        X.append(df.iloc[i-n_timesteps:i, :].values)
        y.append(df.iloc[i, target_col_index])
    return np.array(X), np.array(y)

# Define the model builder function for Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units', min_value=32, max_value=256, step=32),
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32)))
    model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        ),
        loss='mean_squared_error'
    )
    return model

# Define the number of timesteps (sequence length) to test
n_timesteps_list = [30, 60, 90, 180, 365]

# Initialize variables to store the best model and its performance
best_model = None
best_mse = float('inf')
best_timesteps = 0
best_hyperparameters = None

# Loop over different timesteps to find the best one
for n_timesteps in n_timesteps_list:
    # Prepare data for LSTM
    X, y = prepare_data_for_lstm(train_data_lstm, train_df.columns.get_loc('orders'), n_timesteps)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the tuner for the current timestep
    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=2,
        directory='my_dir',
        project_name=f'lstm_tuning_{n_timesteps}'
    )

    # Search for the best hyperparameters
    tuner.search(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    # Get the best model and hyperparameters for the current timestep
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    # Train the best model
    model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), verbose=0)

    # Evaluate the model
    predictions = model.predict(X_val)
    mse = tf.keras.losses.MeanSquaredError()(y_val, predictions).numpy()

    # Store the best model and its performance
    if mse < best_mse:
        best_mse = mse
        best_model = model
        best_timesteps = n_timesteps
        best_hyperparameters = best_hps

print(f"Best number of timesteps: {best_timesteps}")
print(f"Best MSE: {best_mse}")
print(f"Best Hyperparameters: {best_hyperparameters}")

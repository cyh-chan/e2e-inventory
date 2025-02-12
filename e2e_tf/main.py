import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_preparation import prepare_data
from model_training import train_model, check_overfitting
from visualization import plot_inventory_levels, plot_demand_forecast
from analysis import analyze_inventory_performance, print_analysis_results
from e2e_tf_model import create_e2e_model, custom_loss
from keras.src.saving import serialization_lib

serialization_lib.enable_unsafe_deserialization()

# Load the DataFrame from the parquet file
data = pd.read_parquet('../e2e_data_generation/inventory_dataset.parquet')

# Define constants
DROPOUT_RATE = 0.01
MAX_SKU_ID = 100
MAX_STORE_ID = 50
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
EPOCHS = 20
horizon = 90  # number of days in review period
train_horizon = 60
test_horizon = horizon - train_horizon
interval = 10  # number of days between consecutive reorder points
forecast_horizon = 5  # number of days in advance to forecast
seq_len = 15
b = 9
h = 1
dynamic_features = ['inventory', 'demand', 'lead_time', 'reorder_point', 'day', "week"]
n_dynamic_features = len(dynamic_features)
target = ["reorder_qty", "demand", "lead_time"]
model_file_name = "e2e.keras"
log_dir = os.path.join("logs", "scalars")

# Prepare the data
train_input_dynamic_final, train_output_final, cate_feature_final, vlt_input, rp_in, initial_stock_in, rp_out, df_out, vlt_out = prepare_data(
    data, dynamic_features, target, seq_len, forecast_horizon, train_horizon, n_dynamic_features
)

# Create and compile the model
model = create_e2e_model(
    seq_len=seq_len,
    n_dyn_fea=len(dynamic_features) - 3,
    n_outputs=forecast_horizon,
    n_dilated_layers=5,
    kernel_size=2,
    n_filters=3,
    dropout_rate=DROPOUT_RATE,
    max_cat_id=[MAX_SKU_ID, MAX_STORE_ID],
)

adam = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss="mse", optimizer=adam)

# Train the model
history = train_model(
    model, train_input_dynamic_final, cate_feature_final, vlt_input, rp_in, initial_stock_in, df_out, rp_out, vlt_out,
    model_file_name, log_dir, EPOCHS, BATCH_SIZE
)

# Check for overfitting
is_overfitting, analysis = check_overfitting(history)
print(f"Overfitting detected: {is_overfitting}")
print(f"Analysis: {analysis['message']}")
print("\nTraining Statistics:")
print(f"Total epochs: {analysis['statistics']['total_epochs']}")
print(f"Best epoch: {analysis['statistics']['best_epoch']}")
print(f"Best loss: {analysis['statistics']['best_loss']:.6f}")
print(f"Final loss: {analysis['statistics']['final_loss']:.6f}")
print(f"Total improvement: {analysis['statistics']['loss_improvement']:.2f}%")

# Analyze inventory performance
results = analyze_inventory_performance(
    inventory_levels=data['inventory'],
    demand=data['demand'],
    holding_cost=h,
    backorder_cost=b,
    time_periods=(0, train_horizon, train_horizon, horizon)
)

# Print analysis results
print_analysis_results(results)

# Plot inventory levels
plot_inventory_levels(data['inventory'], train_horizon, horizon)

# Generate forecasted demand using the trained model
forecasted_demand = model.predict([train_input_dynamic_final[:, :, (1, 4, 5)], cate_feature_final, vlt_input, rp_in, initial_stock_in])

# Extract the demand forecast from the model's output
forecasted_demand = forecasted_demand[0]  # Adjust this index based on your model's output structure

# Ensure forecasted_demand has the correct shape
forecasted_demand = forecasted_demand[:horizon - train_horizon]  # Truncate to match the testing period

# Plot actual vs. forecasted demand
plot_demand_forecast(data['demand'], forecasted_demand, train_horizon, horizon)

if __name__ == "__main__":
    # Load trained model
    model = load_model(model_file_name, custom_objects={"custom_loss": custom_loss})
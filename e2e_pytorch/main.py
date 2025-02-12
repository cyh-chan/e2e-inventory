import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from invutils.invutils import get_demand, get_lead_time, get_safety_stock, get_reorder_point, get_target_inventory
from e2e_torch_model import E2EModel, custom_loss
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from scipy import stats
import statsmodels.api as sm
import time
from data_preparation import load_data, prepare_data, create_data_loaders
from model_training import train_model
from evaluation import evaluate_model
from visualization import plot_training_loss, prepare_forecast_data, plot_inventory_level, plot_demand_forecast
from analysis import analyze_inventory_performance, print_analysis_results
from utils import check_overfitting


# Load the DataFrame from the parquet file
data = pd.read_parquet('../e2e_data_generation/inventory_dataset.parquet')
column_headers = data.columns.tolist()
# print("Column Headers:", column_headers)

inventory_level = data['inventory']
future_demand = data['demand']
lead_time = data['lead_time']
reorder_point_arr = data['reorder_point']
optimal_rq = data['reorder_qty']
week = data['week']
sku = data['sku']
store = data['store']
data['sku'] = data['sku'].astype(int)
data['store'] = data['store'].astype(int)

DROPOUT_RATE = 0.01
MAX_SKU_ID = 100
MAX_STORE_ID = 50
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
EPOCHS = 20

horizon = 90  # number of days in review period (consider x days in the future for optimal inventory position)
train_horizon = 60
test_horizon = horizon - train_horizon
interval = 10  # number of days between consecutive reorder points
forecast_horizon = 5  # number of days in advance to forecast (at dt, forecast demand from dt + 1 to dt + 5)
seq_len = 15
b = 9
h = 1
week = [i % 7 for i in range(horizon)]
dynamic_features = ['inventory', 'demand', 'lead_time', 'reorder_point', 'day', "week"]
n_dynamic_features = len(dynamic_features)
target = ["reorder_qty", "demand", "lead_time"]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare data
train_input_dynamic_final, train_output_final, cate_feature_final, vlt_input, rp_in, initial_stock_in, rp_out, df_out, vlt_out = prepare_data(
    data, dynamic_features, target, seq_len, forecast_horizon, train_horizon, len(dynamic_features)
)

# Create data loaders
train_loader, val_loader = create_data_loaders(
    train_input_dynamic_final, cate_feature_final, vlt_input, rp_in, initial_stock_in, df_out, rp_out, vlt_out, BATCH_SIZE
)
# Initialize model, optimizer, and loss function
model = E2EModel(
    seq_len=seq_len,
    n_dyn_fea=len(dynamic_features),
    n_outputs=forecast_horizon,
    n_dilated_layers=5,
    kernel_size=2,
    n_filters=3,
    dropout_rate=DROPOUT_RATE,
    max_cat_id=[MAX_SKU_ID, MAX_STORE_ID]
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train model
train_losses, df_pred = train_model(model, train_loader, val_loader, optimizer, device, EPOCHS, custom_loss)

# Evaluate model
test_loss = evaluate_model(model, val_loader, device, custom_loss)

# Check for overfitting
is_overfitting, analysis = check_overfitting(train_losses)
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

print_analysis_results(results)

# Prepare and plot inventory level
train_time, train_inventory, test_time, test_inventory, train_demand, test_demand, forecasted_demand_reshaped = prepare_forecast_data(
    data, df_pred, train_horizon, horizon
)
# Plot inventory level
plot_inventory_level(train_time, train_inventory, test_time, test_inventory)

# Plot demand forecast
plot_demand_forecast(train_time, train_demand, test_time, test_demand, forecasted_demand_reshaped)
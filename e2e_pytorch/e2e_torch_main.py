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

def prepare_data(data, dynamic_features, target, seq_len, forecast_horizon, train_horizon, n_dynamic_features):
    # Initialize lists to store the final training data
    train_input_dynamic_ = []
    train_output_ = []
    cate_feature_ = []

    # Group the data by 'sku' and 'store'
    grouped = data.groupby(['sku', 'store'])

    for (sku, store), group in grouped:
        # Extract the relevant features and target for the current group
        cate_feature = group[['sku', 'store']].values
        arr = group[dynamic_features].values
        output_arr = np.squeeze(group[target].values)

        # Calculate the number of sequences
        n = train_horizon - seq_len - forecast_horizon + 1 - 5

        # Prepare the dynamic input features
        train_input_dynamic = np.empty((n, seq_len, n_dynamic_features))
        for i in range(n):
            train_input_dynamic[i, :, :] = arr[i:i + seq_len]

        # Prepare the output targets
        train_output = np.zeros((n, forecast_horizon, len(target)))
        for i in range(len(train_output)):
            train_output[i, :] = output_arr[i + seq_len: i + seq_len + forecast_horizon, :]

        # Append the prepared data to the lists
        train_input_dynamic_.append(train_input_dynamic)
        train_output_.append(train_output)
        cate_feature_.append(cate_feature[:n, :])

    # Concatenate all the data from different groups
    train_output_final = np.concatenate(train_output_)
    train_input_dynamic_final = np.concatenate(train_input_dynamic_)
    cate_feature_final = np.concatenate(cate_feature_)

    # Extract specific features from the final dynamic input
    vlt_input = train_input_dynamic_final[:, :, 2]
    rp_in = train_input_dynamic_final[:, :, 3]
    initial_stock_in = train_input_dynamic_final[:, :, 0]

    # Extract specific targets from the final output
    rp_out = train_output_final[:, :, 0][:, 0]  # replenishment decision (use history seq_len points to predict 1)
    df_out = train_output_final[:, :, 1]  # demand forecast (use history seq_len points to predict future forecast_horizon)
    vlt_out = train_output_final[:, :, 2][:, 0]

    # Convert to PyTorch tensors
    train_input_dynamic_final = torch.tensor(train_input_dynamic_final, dtype=torch.float32)
    train_output_final = torch.tensor(train_output_final, dtype=torch.float32)
    cate_feature_final = torch.tensor(cate_feature_final, dtype=torch.long)
    vlt_input = torch.tensor(vlt_input, dtype=torch.float32)
    rp_in = torch.tensor(rp_in, dtype=torch.float32)
    initial_stock_in = torch.tensor(initial_stock_in, dtype=torch.float32)
    rp_out = torch.tensor(rp_out, dtype=torch.float32)
    df_out = torch.tensor(df_out, dtype=torch.float32)
    vlt_out = torch.tensor(vlt_out, dtype=torch.float32)

    return train_input_dynamic_final, train_output_final, cate_feature_final, vlt_input, rp_in, initial_stock_in, rp_out, df_out, vlt_out

# Prepare data (assuming prepare_data is already defined)
train_input_dynamic_final, train_output_final, cate_feature_final, vlt_input, rp_in, initial_stock_in, rp_out, df_out, vlt_out = prepare_data(
    data, dynamic_features, target, seq_len, forecast_horizon, train_horizon, n_dynamic_features
)

# Split data into training and validation sets
train_size = int(0.8 * len(train_input_dynamic_final))
val_size = len(train_input_dynamic_final) - train_size

train_dataset = TensorDataset(
    train_input_dynamic_final[:train_size],
    cate_feature_final[:train_size],
    vlt_input[:train_size],
    rp_in[:train_size],
    initial_stock_in[:train_size],
    df_out[:train_size],
    rp_out[:train_size],
    vlt_out[:train_size]
)

val_dataset = TensorDataset(
    train_input_dynamic_final[train_size:],
    cate_feature_final[train_size:],
    vlt_input[train_size:],
    rp_in[train_size:],
    initial_stock_in[train_size:],
    df_out[train_size:],
    rp_out[train_size:],
    vlt_out[train_size:]
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

# Training loop
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        # Unpack batch
        seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in, df_true, rp_true, vlt_true = batch
        seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in = seq_in.to(device), cat_fea_in.to(device), vlt_in.to(device), rp_in.to(device), initial_stock_in.to(device)
        df_true, rp_true, vlt_true = df_true.to(device), rp_true.to(device), vlt_true.to(device)

        # Forward pass
        df_pred, layer4_pred, vlt_pred = model(seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in)

        # Compute loss
        loss = custom_loss((df_true, rp_true, vlt_true), (df_pred, layer4_pred, vlt_pred))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Print average loss for the epoch
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)  # Store the loss for this epoch
    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            # Unpack batch
            seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in, df_true, rp_true, vlt_true = batch
            seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in = seq_in.to(device), cat_fea_in.to(device), vlt_in.to(device), rp_in.to(device), initial_stock_in.to(device)
            df_true, rp_true, vlt_true = df_true.to(device), rp_true.to(device), vlt_true.to(device)

            # Forward pass
            df_pred, layer4_pred, vlt_pred = model(seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in)

            # Compute loss
            loss = custom_loss((df_true, rp_true, vlt_true), (df_pred, layer4_pred, vlt_pred))
            val_loss += loss.item()

    # Print average validation loss
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Validation Loss: {avg_val_loss:.4f}")

# Evaluate on test data
model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch in val_loader:
        # Unpack batch
        seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in, df_true, rp_true, vlt_true = batch
        seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in = seq_in.to(device), cat_fea_in.to(device), vlt_in.to(device), rp_in.to(device), initial_stock_in.to(device)
        df_true, rp_true, vlt_true = df_true.to(device), rp_true.to(device), vlt_true.to(device)

        # Forward pass
        df_pred, layer4_pred, vlt_pred = model(seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in)

        # Compute loss
        loss = custom_loss((df_true, rp_true, vlt_true), (df_pred, layer4_pred, vlt_pred))
        test_loss += loss.item()

# Print average test loss
avg_test_loss = test_loss / len(val_loader)
print(f"Test Loss: {avg_test_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "../e2e_model.pth")

# Load the model
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
model.load_state_dict(torch.load("../e2e_model.pth"))
model.eval()

# 1. Plot Training Loss Over Epochs
def plot_training_loss(train_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# Extract inventory and demand data
train_inventory = inventory_level[:train_horizon + 1]  # Training inventory (first 60 days)
test_inventory = inventory_level[train_horizon:horizon]  # Testing inventory (last 30 days)

train_demand = future_demand[:train_horizon + 1]  # Training demand (first 60 days)
test_demand = future_demand[train_horizon:horizon]  # Testing demand (last 30 days)

# Forecasted demand (assuming df_pred contains forecasted demand for the test horizon)
forecasted_demand = df_pred.cpu().numpy()  # Convert to numpy array if it's a tensor

# Create time axis for training and testing
train_time = np.arange(0, train_horizon + 1)  # Time axis for training data (0 to 59 days)
test_time = np.arange(train_horizon, horizon)  # Time axis for testing data (60 to 89 days)

# Plot Inventory Level Over Time
def plot_inventory_level(train_time, train_inventory, test_time, test_inventory):
    plt.figure(figsize=(10, 6))
    plt.plot(train_time, train_inventory, label='Training Inventory Level', color='blue')

    plt.plot(test_time, test_inventory, label='Testing Inventory Level', color='red')
    plt.xlabel('Time (Days)')
    plt.ylabel('Inventory Level')
    plt.title('Inventory Level Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot Demand Forecast
def plot_demand_forecast(train_time, train_demand, test_time, test_demand, forecasted_demand):
    plt.figure(figsize=(10, 6))
    plt.plot(train_time, train_demand, label='Actual Demand (Training)', color='blue')

    plt.plot(test_time, test_demand, label='Actual Demand (Testing)', color='red')
    plt.plot(test_time, forecasted_demand, label='Forecasted Demand (Testing)', linestyle='--')
    plt.xlabel('Time (Days)')
    plt.ylabel('Demand')
    plt.title('Demand Forecast Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def check_overfitting(train_losses, save_plots=True):
    # Extract loss values
    train_loss = train_losses
    epochs = range(1, len(train_loss) + 1)

    # Calculate moving averages to smooth out fluctuations
    window = 5
    train_ma = np.convolve(train_loss, np.ones(window) / window, mode='valid')

    # Define overfitting criteria
    min_epochs = 10
    sustained_period = 5  # Number of epochs to confirm trend

    # Initialize analysis results
    is_overfitting = False
    analysis = {
        'train_loss': train_loss,
        'train_loss_trend': np.diff(train_ma[-sustained_period:]).mean(),
        'min_loss': min(train_loss),
        'final_loss': train_loss[-1],
        'improvement_rate': None
    }

    # Calculate recent improvement rate
    if len(train_loss) >= sustained_period:
        recent_improvement = (train_loss[-sustained_period] - train_loss[-1]) / train_loss[-sustained_period]
        analysis['improvement_rate'] = recent_improvement

    # Check for overfitting conditions
    if len(train_loss) >= min_epochs:
        # Condition 1: Loss plateauing
        if analysis['improvement_rate'] is not None and analysis['improvement_rate'] < 0.001:
            is_overfitting = True
            analysis['message'] = 'Training has plateaued with minimal improvement'

        # Condition 2: Loss increasing in recent epochs
        elif analysis['train_loss_trend'] > 0:
            is_overfitting = True
            analysis['message'] = 'Training loss is increasing'

        # Condition 3: Significant deviation from minimum loss
        elif (train_loss[-1] - analysis['min_loss']) / analysis['min_loss'] > 0.1:
            is_overfitting = True
            analysis['message'] = 'Current loss significantly higher than best achieved'

        else:
            analysis['message'] = 'No clear signs of overfitting detected'

    else:
        analysis['message'] = 'Not enough epochs to determine overfitting'

    # Add training statistics
    analysis['statistics'] = {
        'total_epochs': len(train_loss),
        'best_epoch': np.argmin(train_loss) + 1,
        'best_loss': min(train_loss),
        'final_loss': train_loss[-1],
        'loss_improvement': (train_loss[0] - train_loss[-1]) / train_loss[0] * 100
    }

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    if save_plots:
        plt.savefig('training_loss.png')
    plt.show()

    return is_overfitting, analysis

def analyze_inventory_performance(inventory_levels, demand, holding_cost, backorder_cost, time_periods=None):
    # Initialize results dictionary
    results = {}

    # Calculate holding and stockout quantities
    positive_inventory = np.maximum(inventory_levels, 0)
    stockouts = np.maximum(-inventory_levels, 0)

    # Calculate costs
    holding_costs = holding_cost * positive_inventory
    stockout_costs = backorder_cost * stockouts
    total_costs = holding_costs + stockout_costs

    # Calculate averages
    results['avg_holding_cost'] = np.mean(holding_costs)
    results['avg_stockout_cost'] = np.mean(stockout_costs)
    results['avg_total_cost'] = np.mean(total_costs)

    # Calculate turnover rate (annual)
    total_demand = np.sum(demand)
    avg_inventory = np.mean(positive_inventory)
    results['turnover_rate'] = (total_demand / avg_inventory) * (365 / len(inventory_levels))

    # Calculate stockout ratio
    stockout_periods = np.sum(stockouts > 0)
    results['stockout_ratio'] = stockout_periods / len(inventory_levels)

    # Perform t-tests
    results['t_tests'] = {}

    for cost_type, costs in [
        ('holding_cost', holding_costs),
        ('stockout_cost', stockout_costs),
        ('total_cost', total_costs)
    ]:
        t_stat, p_value = stats.ttest_1samp(costs, 0)
        results['t_tests'][cost_type] = {
            't_statistic': t_stat,
            'p_value': p_value
        }

    # Perform OLS regression
    time_index = np.arange(len(inventory_levels))

    X = sm.add_constant(np.column_stack((
        time_index,
        demand,
        positive_inventory
    )))

    results['regression'] = {}

    for cost_type, y in [
        ('holding_cost', holding_costs),
        ('stockout_cost', stockout_costs),
        ('total_cost', total_costs)
    ]:
        model = sm.OLS(y, X)
        regression = model.fit()
        results['regression'][cost_type] = {
            'params': regression.params,
            'r_squared': regression.rsquared,
            'adj_r_squared': regression.rsquared_adj,
            'f_stat': regression.fvalue,
            'f_pvalue': regression.f_pvalue,
            'summary': regression.summary()
        }

    # If time periods provided, calculate comparative statistics
    if time_periods:
        train_start, train_end, test_start, test_end = time_periods
        results['comparative'] = {
            'train': {
                'avg_holding_cost': np.mean(holding_costs[train_start:train_end]),
                'avg_stockout_cost': np.mean(stockout_costs[train_start:train_end]),
                'avg_total_cost': np.mean(total_costs[train_start:train_end]),
                'turnover_rate': (np.sum(demand[train_start:train_end]) /
                                  np.mean(positive_inventory[train_start:train_end])) *
                                 (365 / (train_end - train_start)),
                'stockout_ratio': np.sum(stockouts[train_start:train_end] > 0) /
                                  (train_end - train_start)
            },
            'test': {
                'avg_holding_cost': np.mean(holding_costs[test_start:test_end]),
                'avg_stockout_cost': np.mean(stockout_costs[test_start:test_end]),
                'avg_total_cost': np.mean(total_costs[test_start:test_end]),
                'turnover_rate': (np.sum(demand[test_start:test_end]) /
                                  np.mean(positive_inventory[test_start:test_end])) *
                                 (365 / (test_end - test_start)),
                'stockout_ratio': np.sum(stockouts[test_start:test_end] > 0) /
                                  (test_end - test_start)
            }
        }

    return results

def print_analysis_results(results):
    print("\n=== Inventory Performance Analysis ===\n")

    print("Cost Metrics:")
    print(f"Average Holding Cost: {results['avg_holding_cost']:.2f}")
    print(f"Average Stockout Cost: {results['avg_stockout_cost']:.2f}")
    print(f"Average Total Cost: {results['avg_total_cost']:.2f}")

    print("\nPerformance Metrics:")
    print(f"Inventory Turnover Rate (annual): {results['turnover_rate']:.2f}")
    print(f"Stockout Ratio: {results['stockout_ratio']:.2%}")

    print("\nStatistical Tests:")
    for cost_type, test_results in results['t_tests'].items():
        print(f"\n{cost_type.replace('_', ' ').title()} T-Test:")
        print(f"t-statistic: {test_results['t_statistic']:.4f}")
        print(f"p-value: {test_results['p_value']:.4f}")

    print("\nRegression Analysis:")
    for cost_type, reg_results in results['regression'].items():
        print(f"\n{cost_type.replace('_', ' ').title()} Regression:")
        print(f"R-squared: {reg_results['r_squared']:.4f}")
        print(f"Adjusted R-squared: {reg_results['adj_r_squared']:.4f}")
        print(f"F-statistic: {reg_results['f_stat']:.4f}")
        print(f"F-statistic p-value: {reg_results['f_pvalue']:.4f}")

    if 'comparative' in results:
        print("\nComparative Analysis:")
        metrics = ['avg_holding_cost', 'avg_stockout_cost', 'avg_total_cost',
                   'turnover_rate', 'stockout_ratio']

        print("\nMetric               Training    Testing     % Change")
        print("-" * 50)
        for metric in metrics:
            train_val = results['comparative']['train'][metric]
            test_val = results['comparative']['test'][metric]
            pct_change = ((test_val - train_val) / train_val) * 100

            if metric == 'stockout_ratio':
                print(f"{metric.replace('_', ' ').title():20} {train_val:.2%}    {test_val:.2%}    {pct_change:+.1f}%")
            else:
                print(f"{metric.replace('_', ' ').title():20} {train_val:.2f}    {test_val:.2f}    {pct_change:+.1f}%")

# Aggregate forecasted demand (take the mean across the forecast horizon)
forecasted_demand_aggregated = forecasted_demand.mean(axis=1)

# Downsample by averaging
samples_per_day = len(forecasted_demand_aggregated) // test_horizon
forecasted_demand_reshaped = forecasted_demand_aggregated[:test_horizon * samples_per_day].reshape(test_horizon, samples_per_day).mean(axis=1)

# Generate the plots
plot_training_loss(train_losses)
plot_inventory_level(train_time, train_inventory, test_time, test_inventory)
plot_demand_forecast(train_time, train_demand, test_time, test_demand, forecasted_demand_reshaped)

# After model training:
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
    inventory_levels=inventory_level,
    demand=future_demand,
    holding_cost=h,
    backorder_cost=b,
    time_periods=(0, train_horizon, train_horizon, horizon)
)

print_analysis_results(results)

# Plot inventory levels over the horizon
plot_inventory_level(train_time, train_inventory, test_time, test_inventory)

# Plot actual vs. forecasted demand
plot_demand_forecast(train_time, train_demand, test_time, test_demand, forecasted_demand_reshaped)
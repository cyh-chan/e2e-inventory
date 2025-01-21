import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import datetime
import matplotlib.pyplot as plt
from invutils import get_demand, get_lead_time, get_safety_stock, get_reorder_point, get_target_inventory
from torch_model import E2EModel, custom_loss
import time
from scipy import stats
import scipy.stats as stats
import statsmodels.api as sm
from torch.utils.tensorboard import SummaryWriter

start = time.perf_counter()
DROPOUT_RATE = 0.01
MAX_SKU_ID = 100
MAX_STORE_ID = 50
BATCH_SIZE = 1024
LEARNING_RATE = 1e-04
EPOCHS = 20

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

horizon = 90  # number of days in review period
train_horizon = 60
test_horizon = horizon - train_horizon
interval = 10  # number of days between consecutive reorder points
forecast_horizon = 5  # number of days in advance to forecast
seq_len = 15
b = 9
h = 1
week = [i % 7 for i in range(horizon)]
dynamic_features = ['inventory', 'demand', 'lead_time', 'reorder_point', 'day', "week"]
n_dynamic_features = len(dynamic_features)
target = ["reorder_qty", "demand", "lead_time"]


# Custom Dataset
class InventoryDataset(Dataset):
    def __init__(self, dynamic_input, categorical_input, vlt_input, rp_input, initial_stock_input,
                 df_output, rp_output, vlt_output):
        self.dynamic_input = torch.FloatTensor(dynamic_input)
        self.categorical_input = torch.LongTensor(categorical_input)
        self.vlt_input = torch.FloatTensor(vlt_input)
        self.rp_input = torch.FloatTensor(rp_input)
        self.initial_stock_input = torch.FloatTensor(initial_stock_input)
        self.df_output = torch.FloatTensor(df_output)
        self.rp_output = torch.FloatTensor(rp_output).view(-1, 1)  # reshape to [batch_size, 1]
        self.vlt_output = torch.FloatTensor(vlt_output).view(-1, 1)  # reshape to [batch_size, 1]

    def __len__(self):
        return len(self.dynamic_input)

    def __getitem__(self, idx):
        return (self.dynamic_input[idx],
                self.categorical_input[idx],
                self.vlt_input[idx],
                self.rp_input[idx],
                self.initial_stock_input[idx],
                self.df_output[idx],
                self.rp_output[idx],
                self.vlt_output[idx])


# Data generation (same as original)
train_input_dynamic_ = []
train_output_ = []
cate_feature_ = []

for sku in range(1, MAX_SKU_ID + 1):
    mu_demand = np.random.randint(20)
    std_demand = 3
    mu_lead_time = np.random.randint(5)
    std_lead_time = 1
    service_level = 0.99
    for store in range(1, MAX_STORE_ID + 1):

        initial_inventory = np.random.randint(1, 150)  # sku initial stock level

        reorder_point_arr = get_reorder_point(train_horizon, interval)
        n_reorder_pts = len(np.where(reorder_point_arr)[0])  # number of reorder points in review period
        history_lead_time = get_lead_time(mu_lead_time, std_lead_time, 50)
        hm, hs, history_demand = get_demand(mu_demand, std_demand, 50)
        future_lead_time = abs(get_lead_time(mu_lead_time, std_lead_time, n_reorder_pts))
        fm, fs, future_demand = get_demand(mu_demand, std_demand, train_horizon)
        safety_stock = get_safety_stock(history_lead_time, history_demand, service_level)
        # e2e recover optimal re-order quantity
        t = 0
        inventory_level = np.zeros(train_horizon)
        inventory_level[0] = initial_inventory
        # generate po reaching
        po_reaching = list(np.where(reorder_point_arr)[0] + np.int32(np.ceil(future_lead_time)))
        if po_reaching[-1] < train_horizon:
            po_reaching.append(po_reaching[-1] + int(np.ceil(np.diff(po_reaching).mean())))
            optimal_rq = np.zeros(train_horizon)
            for i in range(1, train_horizon):
                inventory_level[i] = inventory_level[i - 1] + inventory_level[i] - future_demand[i]
                if reorder_point_arr[i] and t <= len(po_reaching):
                    vm = po_reaching[t]
                    delta_t = int(np.floor(b * (po_reaching[t + 1] - po_reaching[t]) / (h + b)))  # s*
                    cum_demand = np.sum(future_demand[vm:vm + delta_t])
                    optimal_qty = max(cum_demand + np.sum(future_demand[i:vm]) - inventory_level[i], 0)
                    inventory_level[vm] += optimal_qty
                    optimal_rq[i] = optimal_qty
                    t += 1

            # inventory level, future_demand, future_lead_time, reorder_point, reorder_
            lead_time = np.zeros(train_horizon)
            lead_time[np.where(reorder_point_arr)[0]] = np.int32(np.ceil(future_lead_time))
            # shift reorder point backward by 1 day
            tmp = np.zeros(train_horizon)
            rp = np.where(reorder_point_arr)[0] - 1
            tmp[rp] = 1
            reorder_point_arr = tmp
            data = pd.DataFrame.from_dict({'inventory': inventory_level,
                                           'demand': future_demand,
                                           'lead_time': lead_time,
                                           'reorder_point': reorder_point_arr,
                                           'reorder_qty': optimal_rq,
                                           'week': week[:train_horizon]})
            data["sku"] = sku
            data["store"] = store
            data["day"] = list(range(1, train_horizon + 1))

            cate_feature = data[['sku', 'store']].values
            arr = data[dynamic_features].values
            output_arr = np.squeeze(data[target].values)
            n = train_horizon - seq_len - forecast_horizon + 1 - 5
            train_input_dynamic = np.empty((n, seq_len, n_dynamic_features))
            for i in range(n):
                train_input_dynamic[i, :, :] = arr[i:i + seq_len]

            train_output = np.zeros((n, forecast_horizon, len(target)))
            for i in range(len(train_output)):
                train_output[i, :] = output_arr[i + seq_len: i + seq_len + forecast_horizon, :]
            train_input_dynamic_.append(train_input_dynamic)
            train_output_.append(train_output)
            cate_feature_.append(cate_feature[:n, :])

train_output_final = np.concatenate(train_output_)
train_input_dynamic_final = np.concatenate(train_input_dynamic_)
cate_feature_final = np.concatenate(cate_feature_)

vlt_input = train_input_dynamic_final[:, :, 2]
rp_in = train_input_dynamic_final[:, :, 3]
initial_stock_in = train_input_dynamic_final[:, :, 0]

rp_out = train_output_final[:, :, 0][:, 0]
df_out = train_output_final[:, :, 1]
vlt_out = train_output_final[:, :, 2][:, 0]

print(f"Running time for data generation is {time.perf_counter() - start}")

# Create dataset and dataloader
dataset = InventoryDataset(
    train_input_dynamic_final[:, :, (1, 4, 5)],
    cate_feature_final,
    vlt_input,
    rp_in,
    initial_stock_in,
    df_out,
    rp_out,
    vlt_out
)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
model = E2EModel(
    seq_len=seq_len,
    n_dyn_fea=len(dynamic_features) - 3,
    n_outputs=forecast_horizon,
    n_dilated_layers=5,
    kernel_size=2,
    n_filters=3,
    dropout_rate=DROPOUT_RATE,
    max_cat_id=[MAX_SKU_ID, MAX_STORE_ID],
).to(device)

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# TensorBoard setup
writer = SummaryWriter(os.path.join("logs", "scalars", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

# List to store loss values
loss_values = []

# Training loop
best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_idx, (dynamic_input, cat_input, vlt_in, rp_in, initial_stock_in,
                    df_target, rp_target, vlt_target) in enumerate(train_loader):
        # Move to device
        dynamic_input = dynamic_input.to(device)
        cat_input = cat_input.to(device)
        vlt_in = vlt_in.to(device)
        rp_in = rp_in.to(device)
        initial_stock_in = initial_stock_in.to(device)
        df_target = df_target.to(device)
        rp_target = rp_target.to(device)
        vlt_target = vlt_target.to(device)

        # Forward pass
        df_pred, rp_pred, vlt_pred = model(dynamic_input, cat_input, vlt_in, rp_in, initial_stock_in)

        # Calculate loss
        loss = custom_loss(
            (df_target, rp_target, vlt_target),
            (df_pred, rp_pred, vlt_pred)
        )

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_values.append(avg_loss)  # Store the average loss for plotting
    writer.add_scalar('Loss/train', avg_loss, epoch)

    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}')

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'e2e_model.pth')

print(f"Training completed in {time.perf_counter() - start:.2f} seconds")
print(f"Final Training Loss: {avg_loss:.4f}")

# Prediction and visualization
model.eval()
with torch.no_grad():
    test_dynamic_input = torch.FloatTensor(train_input_dynamic_final[-test_horizon:, :, (1, 4, 5)]).to(device)
    test_cat_input = torch.LongTensor(cate_feature_final[-test_horizon:]).to(device)
    test_vlt_input = torch.FloatTensor(vlt_input[-test_horizon:]).to(device)
    test_rp_input = torch.FloatTensor(rp_in[-test_horizon:]).to(device)
    test_initial_stock_input = torch.FloatTensor(initial_stock_in[-test_horizon:]).to(device)

    df_pred, optimal_qty_pred, vlt_pred = model(
        test_dynamic_input, test_cat_input, test_vlt_input,
        test_rp_input, test_initial_stock_input
    )

    # Convert predictions to numpy for plotting
    df_pred = df_pred.cpu().numpy()
    optimal_qty_pred = optimal_qty_pred.cpu().numpy()
    vlt_pred = vlt_pred.cpu().numpy()


def create_training_plots(loss_values, df_pred, optimal_qty_pred, vlt_pred, df_out, rp_out, vlt_out):
    # Training Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_values) + 1), loss_values, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Demand Forecast vs Actual
    plt.figure(figsize=(10, 6))
    time_steps = range(len(df_pred))
    plt.plot(time_steps, df_pred[:, 0], 'r-', label='Predicted Demand')
    plt.plot(time_steps, df_out[:len(df_pred), 0], 'b-', label='Actual Demand')
    plt.xlabel('Time Step')
    plt.ylabel('Demand')
    plt.title('Demand Forecast vs Actual')
    plt.grid(True)
    plt.legend()
    plt.show()


def create_error_analysis(df_pred, df_out):
    """Create a plot showing prediction errors over time"""
    error_matrix = np.abs(df_pred - df_out[:len(df_pred)])
    mean_error = np.mean(error_matrix, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(mean_error)), mean_error, 'r-', label='Mean Absolute Error')
    plt.xlabel('Time Step')
    plt.ylabel('Mean Absolute Error')
    plt.title('Prediction Error Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


# Use the functions like this:
create_training_plots(loss_values, df_pred, optimal_qty_pred, vlt_pred, df_out, rp_out, vlt_out)
plt.show()


create_error_analysis(df_pred, df_out)
plt.show()


def plot_inventory_levels(inventory_level, train_horizon, horizon, h, b, future_demand, future_lead_time,
                          forecasted_inventory=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get the full range of days
    days = np.arange(len(inventory_level))

    # Plot training inventory levels (up to train_horizon)
    plt.plot(days[:train_horizon],
             inventory_level[:train_horizon],
             label="Historical Inventory Level",
             color='blue')

    # Plot forecasted inventory levels if provided
    if forecasted_inventory is not None:
        forecast_days = np.arange(train_horizon, train_horizon + len(forecasted_inventory))
        plt.plot(forecast_days,
                 forecasted_inventory,
                 label="Forecasted Inventory Level",
                 color='orange',
                 linestyle='-')

    # Add axis labels
    plt.xlabel("Days")
    plt.ylabel("Inventory Level")
    plt.title("E2E Policy DCNN Backbone Result")
    plt.legend()

    # Create statistics text box
    text_str = (f"Holding Cost: {h}\n"
                f"Backorder Cost: {b}\n"
                f"Demand Mean: {future_demand.mean():.2f}, Std: {future_demand.std():.2f}\n"
                f"VLT Mean: {future_lead_time.mean():.2f}, Std: {future_lead_time.std():.2f}")

    # Create text box with wheat color background
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.6, 0.1, text_str,
            horizontalalignment='left',
            fontsize=8,
            verticalalignment='center',
            bbox=props,
            transform=ax.transAxes)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_demand_forecast(future_demand, df_pred, train_horizon):
    """
    Create demand forecast visualization
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get the full range of days
    days = np.arange(len(future_demand))

    # Plot actual demand
    plt.plot(days, future_demand, label='Actual Demand')

    # Plot forecasted demand (starting from train_horizon)
    if df_pred is not None and len(df_pred) > 0:
        forecast_days = np.arange(train_horizon, train_horizon + len(df_pred))
        plt.plot(forecast_days, df_pred[:, 0], label='Forecasted Demand')

    plt.title("Demand Forecast Result with DCNN Backbone")
    plt.xlabel("Days")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Calculate forecasted inventory levels
    forecasted_inventory = np.zeros(len(df_pred))
    current_inventory = inventory_level[train_horizon - 1]  # Start with last known inventory

    for t in range(len(df_pred)):
        # Update inventory based on predicted demand and any predicted orders
        forecasted_inventory[t] = current_inventory - df_pred[t, 0]
        if t < len(optimal_qty_pred):
            forecasted_inventory[t] += optimal_qty_pred[t]
        current_inventory = forecasted_inventory[t]

    # Create and display all plots
    try:
        # Plot 1: Inventory Levels with forecasted inventory
        fig1 = plot_inventory_levels(
            inventory_level=inventory_level,
            train_horizon=train_horizon,
            horizon=horizon,
            h=h,
            b=b,
            future_demand=future_demand,
            future_lead_time=future_lead_time,
            forecasted_inventory=forecasted_inventory
        )
        plt.show()

        # Plot 2: Demand Forecast
        fig2 = plot_demand_forecast(
            future_demand=future_demand,
            df_pred=df_pred,
            train_horizon=train_horizon
        )
        plt.show()

    except Exception as e:
        print(f"Error in plotting: {str(e)}")
        print("Shape information:")
        print(
            f"inventory_level shape: {inventory_level.shape if hasattr(inventory_level, 'shape') else len(inventory_level)}")
        print(f"future_demand shape: {future_demand.shape if hasattr(future_demand, 'shape') else len(future_demand)}")
        print(f"df_pred shape: {df_pred.shape if hasattr(df_pred, 'shape') else len(df_pred)}")

    # Load trained model
    model = E2EModel(
        seq_len=seq_len,
        n_dyn_fea=len(dynamic_features) - 3,
        n_outputs=forecast_horizon,
        n_dilated_layers=5,
        kernel_size=2,
        n_filters=3,
        dropout_rate=DROPOUT_RATE,
        max_cat_id=[MAX_SKU_ID, MAX_STORE_ID],
    )
    model.load_state_dict(torch.load('e2e_model.pth'))
    model.eval()
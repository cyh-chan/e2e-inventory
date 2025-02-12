import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import os
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from invutils.invutils import get_demand, get_lead_time, get_safety_stock, get_reorder_point, get_target_inventory
from model import create_e2e_model, custom_loss
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.src.saving import serialization_lib
import time
from scipy import stats
import scipy.stats as stats
import statsmodels.api as sm

serialization_lib.enable_unsafe_deserialization()

# Load the DataFrame from the parquet file
data = pd.read_parquet('inventory_dataset.parquet')
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

# Print the first few rows of the dataset to ensure it has been loaded correctly
# print("First few rows of the dataset:")
# print(data.head())
# print("\nLast few rows of the dataset:")
# print(data.tail())
# print("Columns in the dataset:")
# print(data.columns)

# print(len(inventory_level))
# print(len(future_demand))
# print(len(lead_time))
# print(len(reorder_point_arr))
# print(len(optimal_rq))
# print(len(sku))
# print(len(store))

start = time.perf_counter()
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

model_file_name = "e2e.keras"
log_dir = os.path.join("logs", "scalars")

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

    return train_input_dynamic_final, train_output_final, cate_feature_final, vlt_input, rp_in, initial_stock_in, rp_out, df_out, vlt_out

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

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss="mse", optimizer=adam)

# Define callbacks
checkpoint = ModelCheckpoint(model_file_name, monitor="loss", save_best_only=True, mode="min", verbose=1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1,
    write_graph=True)
callbacks_list = [checkpoint, tensorboard_callback]

# Train the model
history = model.fit(
    [train_input_dynamic_final[:, :, (1, 4, 5)], cate_feature_final, vlt_input, rp_in, initial_stock_in],
    [df_out, rp_out, vlt_out],
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_list,
    verbose=1,
)

# After model training, print loss and accuracy statistics
print(f"Training completed in {time.perf_counter() - start:.2f} seconds")

# Print the final training loss
final_loss = history.history['loss'][-1]  # Get the last loss value
print(f"Final Training Loss: {final_loss}")

# Optionally, if you are tracking other metrics like accuracy (if applicable), you can print them as well
if 'accuracy' in history.history:
    final_accuracy = history.history['accuracy'][-1]
    print(f"Final Training Accuracy: {final_accuracy:.4f}")

# Plot the loss curve during training
plt.plot(history.history['loss'])
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

def check_overfitting(history, save_plots=True):

    # Extract loss values
    train_loss = history.history['loss']
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

    # Create detailed loss plot
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=1)
    plt.plot(epochs[window - 1:], train_ma, 'r-', label='Moving Average', linewidth=2)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    if save_plots:
        plt.savefig("training_loss.png")

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

    plt.close()
    return is_overfitting, analysis


# Example usage with your existing code:
# After model training:
is_overfitting, analysis = check_overfitting(history)

print(f"Overfitting detected: {is_overfitting}")
print(f"Analysis: {analysis['message']}")
print("\nTraining Statistics:")
print(f"Total epochs: {analysis['statistics']['total_epochs']}")
print(f"Best epoch: {analysis['statistics']['best_epoch']}")
print(f"Best loss: {analysis['statistics']['best_loss']:.6f}")
print(f"Final loss: {analysis['statistics']['final_loss']:.6f}")
print(f"Total improvement: {analysis['statistics']['loss_improvement']:.2f}%")


def plot_inventory_levels(inventory_levels, train_horizon, horizon):
    plt.figure(figsize=(12, 6))

    # # Plot training inventory
    # plt.plot(range(train_horizon), inventory_levels[:train_horizon], label='Training Inventory', color='blue')
    #
    # # Plot testing inventory
    # plt.plot(range(train_horizon, horizon), inventory_levels[train_horizon:horizon], label='Testing Inventory',
    #          color='red')

    # Plot training inventory
    plt.plot(range(train_horizon), inventory_levels[:train_horizon], label='Training Inventory', color='blue')

    # Connect last training point to first testing point
    plt.plot([train_horizon - 1, train_horizon],
             [inventory_levels[train_horizon - 1], inventory_levels[train_horizon]],
             color='blue')

    # Plot testing inventory
    plt.plot(range(train_horizon, horizon), inventory_levels[train_horizon:horizon], label='Testing Inventory', color='red')

    # Add vertical line to separate training and testing
    plt.axvline(x=train_horizon, color='black', linestyle='--', label='Train/Test Split')

    plt.title('Inventory Levels Over Horizon (Training vs. Testing)')
    plt.xlabel('Time (Days)')
    plt.ylabel('Inventory Level')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_demand_forecast(actual_demand, forecasted_demand, train_horizon, horizon):
    plt.figure(figsize=(12, 6))

    # Plot actual demand
    plt.plot(range(horizon), actual_demand[:horizon], label='Actual Demand', color='blue')

    # Plot forecasted demand (only for the testing period)
    testing_period = range(train_horizon, horizon)

    # Ensure forecasted_demand is 1D and matches the testing period length
    if forecasted_demand.ndim > 1:
        print(f"Warning: forecasted_demand has shape {forecasted_demand.shape}. Reshaping to 1D.")
        forecasted_demand = forecasted_demand.flatten()  # Convert to 1D array

    if len(forecasted_demand) != len(testing_period):
        print(
            f"Warning: forecasted_demand length ({len(forecasted_demand)}) does not match testing period length ({len(testing_period)}). Truncating or padding.")
        forecasted_demand = forecasted_demand[:len(testing_period)]  # Truncate if longer
        if len(forecasted_demand) < len(testing_period):
            forecasted_demand = np.pad(forecasted_demand, (0, len(testing_period) - len(forecasted_demand)),
                                       mode='constant')  # Pad if shorter

    plt.plot(testing_period, forecasted_demand, label='Forecasted Demand', color='red', linestyle='--')

    # Add vertical line to separate training and testing
    plt.axvline(x=train_horizon, color='black', linestyle='--', label='Train/Test Split')

    plt.title('Actual vs. Forecasted Demand')
    plt.xlabel('Time (Days)')
    plt.ylabel('Demand')
    plt.legend()
    plt.grid(True)
    plt.show()

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
    # H0: mean cost = 0 vs H1: mean cost ≠ 0
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
    # Create time index
    time_index = np.arange(len(inventory_levels))

    # Prepare data for regression
    X = sm.add_constant(np.column_stack((
        time_index,
        demand,
        positive_inventory
    )))

    # Run regression for each cost type
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


# Example usage:
# With your existing variables:
results = analyze_inventory_performance(
    inventory_levels=inventory_level,
    demand=future_demand,
    holding_cost=h,
    backorder_cost=b,
    time_periods=(0, train_horizon, train_horizon, horizon)
)

print_analysis_results(results)

# Plot inventory levels over the horizon
plot_inventory_levels(inventory_level, train_horizon, horizon)

# Generate forecasted demand using the trained model
forecasted_demand = model.predict([train_input_dynamic_final[:, :, (1, 4, 5)], cate_feature_final, vlt_input, rp_in, initial_stock_in])

# Extract the demand forecast from the model's output
forecasted_demand = forecasted_demand[0]  # Adjust this index based on your model's output structure

# Ensure forecasted_demand has the correct shape
forecasted_demand = forecasted_demand[:horizon - train_horizon]  # Truncate to match the testing period

# Plot actual vs. forecasted demand
plot_demand_forecast(future_demand, forecasted_demand, train_horizon, horizon)

if __name__ == "__main__":
    # Load trained model
    model_file_name = "e2e.keras"
    model = load_model(model_file_name, custom_objects={"custom_loss": custom_loss})
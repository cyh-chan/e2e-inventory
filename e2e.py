import numpy as np
import pandas as pd
import os
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from invutils import get_demand, get_lead_time, get_safety_stock, get_reorder_point, get_target_inventory
from model import create_e2e_model, custom_loss
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.src.saving import serialization_lib
import time
import onnx
import onnxruntime
import tf2onnx
from scipy import stats
import scipy.stats as stats
import statsmodels.api as sm

serialization_lib.enable_unsafe_deserialization()

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

rp_out = train_output_final[:, :, 0][:, 0]  # replenishment decision (use history seq_len points to predict 1)
df_out = train_output_final[:, :, 1]  # demand forecast (use history seq_len points to predict future forecast_horizon)
vlt_out = train_output_final[:, :, 2][:, 0]  # vendor lead time forecast (use history seq_len points to predict 1)
print(f"Running time for data generation is {time.perf_counter() - start}")
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
sgd = optimizers.SGD()
model.compile(loss="mse", optimizer=adam)
# model.compile(
#     optimizer='adam',
#     loss=custom_loss,
# )
# Define checkpoint and fit model
checkpoint = ModelCheckpoint(model_file_name, monitor="loss", save_best_only=True, mode="min", verbose=1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1,
    write_graph=True)
callbacks_list = [checkpoint, tensorboard_callback]
history = model.fit(
    [train_input_dynamic_final[:, :, (1, 4, 5)], cate_feature_final, vlt_input, rp_in, initial_stock_in],
    [df_out, rp_out, vlt_out],
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_list,
    verbose=1,
)

# get a brand new sku
initial_inventory = 10  # sku initial stock level

reorder_point_arr = get_reorder_point(horizon, interval)
n_reorder_pts = len(np.where(reorder_point_arr)[0])  # number of reorder points in review period
history_lead_time = get_lead_time(mu_lead_time, std_lead_time, 50)
hm, hs, history_demand = get_demand(mu_demand, std_demand, 50)
future_lead_time = abs(get_lead_time(mu_lead_time, std_lead_time, n_reorder_pts))
fm, fs, future_demand = get_demand(mu_demand, std_demand, horizon)
safety_stock = get_safety_stock(history_lead_time, history_demand, service_level)
# e2e recover optimal re-order quantity
t = 0
inventory_level = np.zeros(horizon)
inventory_level[0] = initial_inventory
# generate po reaching
po_reaching = list(np.where(reorder_point_arr)[0] + np.int32(np.ceil(future_lead_time)))

po_reaching.append(po_reaching[-1] + int(np.ceil(np.diff(po_reaching).mean())))
optimal_rq = np.zeros(horizon)
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
lead_time = np.zeros(horizon)
lead_time[np.where(reorder_point_arr)[0]] = np.int32(np.ceil(future_lead_time))
# shift reorder point backward by 1 day
tmp = np.zeros(horizon)
rp = np.where(reorder_point_arr)[0] - 1
tmp[rp] = 1
reorder_point_arr = tmp
data = pd.DataFrame.from_dict({'inventory': inventory_level,
                               'demand': future_demand,
                               'lead_time': lead_time,
                               'reorder_point': reorder_point_arr,
                               'reorder_qty': optimal_rq,
                               'week': week[:horizon]})
data["sku"] = sku
data["store"] = store
data["day"] = list(range(1, horizon + 1))

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

train_input_dynamic_final = train_input_dynamic
cate_feature_final = cate_feature[:n, :]
vlt_input = train_input_dynamic_final[:, :, 2]
rp_in = train_input_dynamic_final[:, :, 3]
initial_stock_in = train_input_dynamic_final[:, :, 0]

pred_len = test_horizon

df_pred, optimal_qty_pred, vlt_pred = model.predict([train_input_dynamic_final[-pred_len:, :, (1, 4, 5)],
                                                     cate_feature_final[-pred_len:], vlt_input[-pred_len:],
                                                     rp_in[-pred_len:], initial_stock_in[-pred_len:]])

ilt = inventory_level[:train_horizon].copy()
it = np.zeros(horizon)
it[:train_horizon] = ilt
optimal_rq = np.concatenate((optimal_rq, np.zeros(test_horizon)))
inventory_level = it
future_lead_time = abs(get_lead_time(future_lead_time.mean(), future_lead_time.std(), n_reorder_pts))
fm, fs, future_demand = get_demand(fm, fs, horizon)
rp_qty = np.where(optimal_qty_pred > 5)[0]
po_reaching = list(np.where(reorder_point_arr)[0] + np.int32(np.ceil(future_lead_time)))
po_reaching = po_reaching[int(train_horizon / interval):]
cnt = 0
t = 0
for i in range(train_horizon, horizon):
    inventory_level[i] = inventory_level[i - 1] + inventory_level[i] - future_demand[i]
    if reorder_point_arr[i] and t <= len(po_reaching):
        if cnt == len(rp_qty):
            break
        vm = po_reaching[t]
        optimal_qty = optimal_qty_pred[rp_qty[cnt]]
        inventory_level[vm] += optimal_qty
        optimal_rq[i] = optimal_qty
        print(vm, optimal_qty)
        t += 1
        cnt += 1

fig, ax = plt.subplots()
plt.plot(inventory_level)
pred_inv = inventory_level.copy()
plt.plot(list(range(train_horizon, horizon)), inventory_level[train_horizon:])
# add axix label
plt.xlabel("Days")
plt.ylabel("Inventory Level")

plt.legend(["Training Inventory Level", "Testing Inventory Level"])
plt.title("E2E policy DCNN backbone Result")

text_str = (f"Holding Cost: {h} \n"
            f"Backorder Cost: {b} \n"
            f"Demand Mean: {future_demand.mean(): .2f} , Std: {future_demand.std():.2f} \n"
            f"VLT Mean: {future_lead_time.mean(): .2f}, Std: {future_lead_time.std(): .2f}")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
ax.text(0.6, 0.1, text_str, horizontalalignment='left', fontsize=8,
        verticalalignment='center', bbox=props, transform=ax.transAxes)
plt.savefig("./E2E result.png")
plt.show()

plt.plot(future_demand)
plt.plot(list(range(train_horizon, horizon)), df_pred[:, 0])
plt.title("Demand Forecast Result with DCNN backbone")
plt.legend(['Actual Demand', 'Forecasted Demand'])
plt.show()

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

def convert_to_onnx(model, output_path, seq_len, n_dynamic_features):
    print("Starting ONNX conversion...")

    # Define input signature
    spec = (tf.TensorSpec((None, seq_len, n_dynamic_features), tf.float32),  # Input dynamic features
            tf.TensorSpec((None, 2), tf.int32),  # Categorical features
            tf.TensorSpec((None, seq_len), tf.float32),  # VLT input
            tf.TensorSpec((None, seq_len), tf.float32),  # Reorder point input
            tf.TensorSpec((None, seq_len), tf.float32))  # Initial stock input

    # Convert the model to ONNX
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

    # Save the ONNX model
    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"Model successfully converted to ONNX format and saved as {output_path}")

    # Validate ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed.")
    print(onnx.helper.printable_graph(onnx_model.graph))

    return onnx_model


def test_onnx_model(onnx_model_path, test_inputs):
    print("Testing ONNX model with ONNX Runtime...")
    session = onnxruntime.InferenceSession(onnx_model_path)

    # Prepare inputs
    input_names = {input.name: test_inputs[i] for i, input in enumerate(session.get_inputs())}

    # Run inference
    outputs = session.run(None, input_names)
    print("ONNX Runtime inference completed successfully.")
    return outputs


if __name__ == "__main__":
    # Load trained model
    model_file_name = "e2e.keras"
    model = load_model(model_file_name, custom_objects={"custom_loss": custom_loss})

'''
    # Define ONNX conversion parameters
    onnx_model_path = "e2e_model.onnx"
    seq_len = 15
    n_dynamic_features = len(dynamic_features) - 3

    # Convert and validate ONNX model
    onnx_model = convert_to_onnx(model, onnx_model_path, seq_len, n_dynamic_features)

    # Prepare inputs for testing ONNX Runtime
    test_inputs = [
        train_input_dynamic_final[:, :, (1, 4, 5)].astype(np.float32),  # Dynamic features
        cate_feature_final.astype(np.int32),                          # Categorical features
        vlt_input.astype(np.float32),                                 # VLT input
        rp_in.astype(np.float32),                                     # Reorder point input
        initial_stock_in.astype(np.float32)                           # Initial stock input
    ]

    # Test ONNX model
    outputs = test_onnx_model(onnx_model_path, test_inputs)
    df_pred, optimal_qty_pred, vlt_pred = outputs

    # Optional: Log or visualize predictions
    # print("Demand forecast predictions:", df_pred)
'''

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

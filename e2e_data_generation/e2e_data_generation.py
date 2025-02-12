import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import datetime
from invutils.invutils import get_demand, get_lead_time, get_safety_stock, get_reorder_point, get_target_inventory
from keras.src.saving import serialization_lib
import time

# def set_random_seed(seed=42):
#     np.random.seed(seed)

# Collect all the data into a single DataFrame
all_data = []

start = time.perf_counter()
DROPOUT_RATE = 0.01
MAX_SKU_ID = 100
MAX_STORE_ID = 50

horizon = 90
train_horizon = 60
test_horizon = horizon - train_horizon
interval = 10
forecast_horizon = 5
seq_len = 15
b = 9
h = 1
week = [i % 7 for i in range(horizon)]
dynamic_features = ['inventory', 'demand', 'lead_time', 'reorder_point', 'day', "week"]
n_dynamic_features = len(dynamic_features)
target = ["reorder_qty", "demand", "lead_time"]

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

            # Prepare lead time and reorder point
            lead_time = np.zeros(train_horizon)
            lead_time[np.where(reorder_point_arr)[0]] = np.int32(np.ceil(future_lead_time))

            # shift reorder point backward by 1 day
            tmp = np.zeros(train_horizon)
            rp = np.where(reorder_point_arr)[0] - 1
            tmp[rp] = 1
            reorder_point_arr = tmp

            # Create a DataFrame for this SKU and store
            data = pd.DataFrame({
                'inventory': inventory_level,
                'demand': future_demand,
                'lead_time': lead_time,
                'reorder_point': reorder_point_arr,
                'reorder_qty': optimal_rq,
                'week': week[:train_horizon],
                'sku': sku,
                'store': store,
                'day': list(range(1, train_horizon + 1))
            })

            all_data.append(data)

# Combine all data into a single DataFrame
final_dataframe = pd.concat(all_data, ignore_index=True)

# Optional: Display basic information about the DataFrame
print(final_dataframe.info())
print("\nFirst few rows:")
print(final_dataframe.head())
# print("\nLast few rows of the dataset:")
# print(final_dataframe.tail())

# Save full dataset to parquet
final_dataframe.to_parquet('inventory_dataset.parquet')

# Save full dataset to CSV
final_dataframe.to_csv('inventory_dataset.csv', index=False)

print(f"Files saved: inventory_full_dataset.parquet and inventory_dataset.csv")
print(f"Total rows saved: {len(final_dataframe)}")

inventory_level = data['inventory']
future_demand = data['demand']
lead_time = data['lead_time']
reorder_point_arr = data['reorder_point']
optimal_rq = data['reorder_qty']
week = data['week']
sku = data['sku']
store = data['store']

print(len(inventory_level))
print(len(future_demand))
print(len(lead_time))
print(len(reorder_point_arr))
print(len(optimal_rq))
print(len(week[:train_horizon]))
print(len(sku))
print(len(store))
print(len(list(range(1, train_horizon + 1))))


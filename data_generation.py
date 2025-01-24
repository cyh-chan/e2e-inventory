import numpy as np
import pandas as pd
import os
from invutils import get_demand, get_lead_time, get_safety_stock, get_reorder_point, get_target_inventory


def generate_inventory_data(
        max_sku_id=100,
        max_store_id=50,
        horizon=90,
        train_horizon=60,
        interval=10,
        seq_len=15,
        forecast_horizon=5,
        dynamic_features=['inventory', 'demand', 'lead_time', 'reorder_point', 'day', 'week'],
        target=["reorder_qty", "demand", "lead_time"],
        b=9,
        h=1
):

    train_input_dynamic_ = []
    train_output_ = []
    cate_feature_ = []

    # Week representation
    week = [i % 7 for i in range(horizon)]
    n_dynamic_features = len(dynamic_features)

    for sku in range(1, max_sku_id + 1):
        mu_demand = np.random.randint(20)
        std_demand = 3
        mu_lead_time = np.random.randint(5)
        std_lead_time = 1
        service_level = 0.99

        for store in range(1, max_store_id + 1):
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

                data = pd.DataFrame.from_dict({
                    'inventory': inventory_level,
                    'demand': future_demand,
                    'lead_time': lead_time,
                    'reorder_point': reorder_point_arr,
                    'reorder_qty': optimal_rq,
                    'week': week[:train_horizon]
                })
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

    # Concatenate all generated data
    train_output_final = np.concatenate(train_output_)
    train_input_dynamic_final = np.concatenate(train_input_dynamic_)
    cate_feature_final = np.concatenate(cate_feature_)

    # Separate inputs and outputs
    vlt_input = train_input_dynamic_final[:, :, 2]
    rp_in = train_input_dynamic_final[:, :, 3]
    initial_stock_in = train_input_dynamic_final[:, :, 0]

    rp_out = train_output_final[:, :, 0][:, 0]
    df_out = train_output_final[:, :, 1]
    vlt_out = train_output_final[:, :, 2][:, 0]

    # Create data dictionary to save
    data_dict = {
        'dynamic_input': train_input_dynamic_final[:, :, (1, 4, 5)],
        'categorical_input': cate_feature_final,
        'vlt_input': vlt_input,
        'rp_input': rp_in,
        'initial_stock_input': initial_stock_in,
        'df_output': df_out,
        'rp_output': rp_out,
        'vlt_output': vlt_out
    }

    return data_dict


def save_generated_data(data_dict, filename='inventory_dataset.npz'):
    np.savez_compressed(filename, **data_dict)
    print(f"Data saved to {filename}")


def main():
    generated_data = generate_inventory_data()
    save_generated_data(generated_data)


if __name__ == "__main__":
    main()
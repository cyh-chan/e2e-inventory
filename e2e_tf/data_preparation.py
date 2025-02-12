# data_preparation.py

import numpy as np
import pandas as pd

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
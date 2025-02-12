import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_data(filepath):
    data = pd.read_parquet(filepath)
    data['sku'] = data['sku'].astype(int)
    data['store'] = data['store'].astype(int)
    return data

def prepare_data(data, dynamic_features, target, seq_len, forecast_horizon, train_horizon, n_dynamic_features):
    train_input_dynamic_ = []
    train_output_ = []
    cate_feature_ = []

    grouped = data.groupby(['sku', 'store'])

    for (sku, store), group in grouped:
        cate_feature = group[['sku', 'store']].values
        arr = group[dynamic_features].values
        output_arr = np.squeeze(group[target].values)

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

def create_data_loaders(train_input_dynamic_final, cate_feature_final, vlt_input, rp_in, initial_stock_in, df_out, rp_out, vlt_out, batch_size):
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
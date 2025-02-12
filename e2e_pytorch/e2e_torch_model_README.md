# End-to-End Inventory Management Model with PyTorch

This document provides a detailed explanation of the `E2EModel`, a neural network model implemented using PyTorch for end-to-end inventory management. The model is designed to predict optimal reorder quantities, forecast demand, and manage inventory levels efficiently. Below, we break down the architecture, layers, and functionality of the model.

## Table of Contents
1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
   - [Embedding Layers](#embedding-layers)
   - [Dilated Convolution Layers](#dilated-convolution-layers)
   - [Final Convolution Layer](#final-convolution-layer)
   - [Dense Layers](#dense-layers)
   - [VLT Processing](#vlt-processing)
   - [Reorder Point and Initial Stock Layers](#reorder-point-and-initial-stock-layers)
3. [Custom Loss Function](#custom-loss-function)
4. [Usage](#usage)

## Overview

The `E2EModel` is a neural network designed to handle sequential and categorical data for inventory management tasks. It uses a combination of dilated convolutions, dense layers, and embedding layers to process input data and make predictions. The model outputs three key predictions:
1. **Demand Forecast (`df_output`)**: Predicts future demand over a specified horizon.
2. **Reorder Quantity (`layer4_output`)**: Predicts the optimal reorder quantity.
3. **Lead Time (`vlt_output`)**: Predicts the lead time for replenishment.

## Model Architecture

### Embedding Layers

The model starts by processing categorical features (i.e. SKU and Store IDs) using embedding layers. These layers convert categorical IDs into dense vectors, which are then used as inputs to the network.

```python
self.embeddings = nn.ModuleList([
    nn.Embedding(m + 1, math.ceil(math.log(m + 1)))
    for m in max_cat_id
])
```
- **Input**: Categorical IDs (i.e. SKU and Store IDs).
- **Output**: Dense vectors representing the categorical features.

### Dilated Convolution Layers
The model uses dilated convolutions to process sequential input data (i.e. inventory levels, demand, lead time). Dilated convolutions allow the model to capture long-range dependencies in the data.
```bash
self.conv_layers = nn.ModuleList()
self.conv_layers.append(
    nn.Conv1d(
        n_dyn_fea,
        n_filters,
        kernel_size,
        padding='same',
        dilation=1
    )
)
for i in range(1, n_dilated_layers):
    self.conv_layers.append(
        nn.Conv1d(
            n_filters,
            n_filters,
            kernel_size,
            padding='same',
            dilation=2 ** i
        )
    )
```
- **Input**: Sequential data (i.e. inventory levels, demand, lead time).
- **Output**: Feature maps capturing temporal patterns in the data.

### Dense Layers
The output from the convolutional layers is flattened and concatenated with the embedded categorical features. This combined data is then passed through dense layers to produce the final predictions.
```bash
self.dense1 = nn.Linear(conv_out_size + embed_total_size, 64)
self.output_layer = nn.Linear(64, n_outputs)
```
- **Input**: Flattened convolutional output and embedded categorical features.
- **Output**: Intermediate dense layer output and final demand forecast.

### VLT Processing
The model processes lead time (`vlt_input`) separately using dense layers. This allows the model to capture the impact of lead time on inventory decisions.
```bash
self.vlt_dense1 = nn.Linear(seq_len + embed_total_size, 16)
self.vlt_dense2 = nn.Linear(16, 1)
```
- **Input**: Lead time data and embedded categorical features.
- **Output**: Predicted lead time.

### Reorder Point and Initial Stock Layers
The model includes additional layers to process reorder points and initial stock levels. These layers help the model make more accurate predictions about when to reorder and how much to reorder.
```bash
self.layer3_dense = nn.Linear(64 + 16 + seq_len, 32)
self.layer4_dense1 = nn.Linear(32 + seq_len, 64)
self.layer4_dense2 = nn.Linear(64, 1)
```
- **Input**: Intermediate outputs from previous layers, reorder points, and initial stock levels.
- **Output**: Predicted reorder quantity.

## Custom Loss Function
The model uses a custom loss function to optimize the predictions. The loss function combines the mean squared error (MSE) for demand forecast, reorder quantity, and lead time predictions.
```bash
def custom_loss(y_true, y_pred):
    df_true, layer4_true, vlt_true = y_true
    df_pred, layer4_pred, vlt_pred = y_pred

    mse_df = F.mse_loss(df_pred, df_true)
    mse_layer4 = F.mse_loss(layer4_pred, layer4_true)
    mse_vlt = F.mse_loss(vlt_pred, vlt_true)

    loss = 0.05 * mse_df + 0.9 * mse_layer4 + 0.05 * mse_vlt
    return loss
```
- **Input**: True and predicted values for demand forecast, reorder quantity, and lead time.
- **Output**: Weighted sum of MSE losses.

## Usage
To use the `E2EModel`, follow these steps:
1. **Initialize the Model**
```bash
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
```

2. **Define the Optimizer:**
```bash
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

3. **Train the Model**
```bash
train_losses, df_pred = train_model(model, train_loader, val_loader, optimizer, device, EPOCHS, custom_loss)
```

4. **Evaluate the Model**
```bash
test_loss = evaluate_model(model, val_loader, device, custom_loss)
```

5. **Visualize the Results**
```bash
plot_inventory_level(train_time, train_inventory, test_time, test_inventory)
plot_demand_forecast(train_time, train_demand, test_time, test_demand, forecasted_demand_reshaped) 
```
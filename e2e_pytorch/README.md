# End-to-End Inventory Management Model with PyTorch

This project implements an end-to-end inventory management model using PyTorch. The model is designed to predict optimal reorder quantities, forecast demand, and manage inventory levels efficiently. The project is structured into multiple modules, including data preparation, model training, evaluation, and visualization.

## Table of Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Visualization](#visualization)

## Requirements

To run this project, you need the following Python libraries:

- `numpy`
- `pandas`
- `torch`
- `matplotlib`
- `scipy`
- `statsmodels`
- `tqdm`

You can install the required libraries using pip:

```bash
pip install numpy pandas torch matplotlib scipy statsmodels tqdm
```

## Installation
1. Clone the repository or download the script to your local machine.
2. Ensure that the `inventory_dataset.parquet` file is available in the specified directory.

## Project Structure
The project is organized into the following modules:
- `main.py`: The main script that orchestrates the execution of the project.
- `analysis.py`: Contains functions for analyzing inventory performance.
- `data_preparation.py`: Handles data preprocessing and preparation.
- `e2e_torch_model.py`: Defines the end-to-end inventory management model using PyTorch.
- `evaluation.py`: Contains functions for evaluating the model's performance.
- `model_training.py`: Handles the training of the model.
- `utils.py`: Utility functions for various tasks.
- `visualization.py`: Functions for visualizing the results.

## Usage
To run the project, execute the `main.py` script:
```bash
python main.py
```

This will perform the following steps:
1. Load the dataset.
2. Prepare the data for training.
3. Train the model.
4. Evaluate the model.
5. Visualize the results.

## Model Training
The model is trained using the following parameters:
1. **Batch Size**: 1024
2. **Learning Rate**: 1e-4
3. **Epochs**: 20 (Placeholder value)
4. **Dropout Rate**: 0.01

The model is compiled using the Adam optimizer and a custom loss function.
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

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

## Evaluation
The model's performance is evaluated using various metrics, including training loss, overfitting analysis, and inventory performance metrics.

**Overfitting Analysis** \
The `check_overfitting` function analyzes the training loss to detect signs of overfitting:
```bash
is_overfitting, analysis = check_overfitting(train_losses)
print(f"Overfitting detected: {is_overfitting}")
print(f"Analysis: {analysis['message']}")
print("\nTraining Statistics:")
print(f"Total epochs: {analysis['statistics']['total_epochs']}")
print(f"Best epoch: {analysis['statistics']['best_epoch']}")
print(f"Best loss: {analysis['statistics']['best_loss']:.6f}")
print(f"Final loss: {analysis['statistics']['final_loss']:.6f}")
print(f"Total improvement: {analysis['statistics']['loss_improvement']:.2f}%")
```

**Inventory Performance Analysis** \
The `analyze_inventory_performance` function calculates various inventory performance metrics, including holding costs, stockout costs, and turnover rates:

```bash
results = analyze_inventory_performance(
    inventory_levels=data['inventory'],
    demand=data['demand'],
    holding_cost=h,
    backorder_cost=b,
    time_periods=(0, train_horizon, train_horizon, horizon)
)

print_analysis_results(results)
```

## Visualization
The project includes several visualization functions to help understand the model's performance:

**Inventory Levels** \
The `plot_inventory_level` function plots the inventory levels over the training and testing horizons:
```bash
plot_inventory_level(train_time, train_inventory, test_time, test_inventory)
```

**Demand Forecast** \
The `plot_demand_forecast` function compares the actual demand with the forecasted demand:
```bash
plot_demand_forecast(train_time, train_demand, test_time, test_demand, forecasted_demand_reshaped)
```
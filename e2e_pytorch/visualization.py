import matplotlib.pyplot as plt
import numpy as np

def plot_training_loss(train_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_training_accuracy(train_accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def prepare_forecast_data(data, df_pred, train_horizon, horizon):
    # Create time axis for training and testing
    train_time = np.arange(0, train_horizon + 1)  # Time axis for training data (0 to 59 days)
    test_time = np.arange(train_horizon, horizon)  # Time axis for testing data (60 to 89 days)

    # Extract inventory and demand data
    train_inventory = data['inventory'][:train_horizon + 1]  # Training inventory (first 60 days)
    test_inventory = data['inventory'][train_horizon:horizon]  # Testing inventory (last 30 days)

    train_demand = data['demand'][:train_horizon + 1]  # Training demand (first 60 days)
    test_demand = data['demand'][train_horizon:horizon]  # Testing demand (last 30 days)

    # Forecasted demand (assuming df_pred contains forecasted demand for the test horizon)
    forecasted_demand = df_pred.cpu().numpy()  # Convert to numpy array if it's a tensor

    # Aggregate forecasted demand (take the mean across the forecast horizon)
    forecasted_demand_aggregated = forecasted_demand.mean(axis=1)

    # Downsample by averaging
    test_horizon = horizon - train_horizon
    samples_per_day = len(forecasted_demand_aggregated) // test_horizon
    forecasted_demand_reshaped = forecasted_demand_aggregated[:test_horizon * samples_per_day].reshape(test_horizon, samples_per_day).mean(axis=1)

    return train_time, train_inventory, test_time, test_inventory, train_demand, test_demand, forecasted_demand_reshaped

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

def plot_demand_forecast(train_time, train_demand, test_time, test_demand, forecasted_demand):
    plt.figure(figsize=(10, 6))
    plt.plot(train_time, train_demand, label='Actual Demand (Training)', color='blue')
    plt.plot(test_time, test_demand, label='Actual Demand (Testing)', color='red')
    plt.plot(test_time, forecasted_demand, label='Forecasted Demand (Testing)', linestyle='--', color='green')
    plt.xlabel('Time (Days)')
    plt.ylabel('Demand')
    plt.title('Demand Forecast Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
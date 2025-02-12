import matplotlib.pyplot as plt
import numpy as np


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
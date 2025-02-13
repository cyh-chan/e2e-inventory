# Inventory Management Simulation and Analysis

This project provides a comprehensive framework for simulating and analyzing inventory management policies. The simulation is designed to evaluate different inventory replenishment strategies under various demand distributions. The project is structured into multiple modules, including `inventory_simulation`, `inventory_visualization`, `inventory_analysis`, and `inventory_policies`, which are orchestrated by the `main.py` script.

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Modules](#modules)
   - [Inventory Simulation](#inventory-simulation)
   - [Inventory Visualization](#inventory-visualization)
   - [Inventory Analysis](#inventory-analysis)
   - [Inventory Policies](#inventory-policies)
4. [Usage](#usage)

## Overview

The `main.py` script serves as the entry point for the project. It simulates inventory management under different demand distributions, visualizes the results, and performs an in-depth analysis of the inventory policies. The project is designed to be modular, with each module handling a specific aspect of the inventory management process.

## Project Structure

The project is organized into the following modules:

- **main.py**: The main script that orchestrates the simulation, visualization, and analysis.
- **inventory_simulation.py**: Handles the simulation of inventory management policies.
- **inventory_visualization.py**: Provides functions for visualizing the simulation results.
- **inventory_analysis.py**: Contains functions for analyzing the performance of inventory policies.
- **inventory_policies.py**: Implements various inventory replenishment policies.

## Modules

### Inventory Simulation

The `InventorySimulation` class in `inventory_simulation.py` is responsible for simulating inventory management under different demand distributions. It uses various inventory policies such as Predict-Then-Optimize (PTO), and End-to-End (E2E) to manage inventory levels.

#### Key Functions:
- **`run_simulation`**: Runs the simulation and returns the results, including inventory levels, purchase orders, and lead times.

### Inventory Visualization

The `InventoryVisualization` class in `inventory_visualization.py` provides functions for visualizing the simulation results. It generates plots to compare the performance of different inventory policies.

#### Key Functions:
- **`plot_simulation`**: Generates plots to visualize inventory levels and purchase orders over time.
- **`export_data`**: Exports the simulation data to a CSV file for further analysis.

### Inventory Analysis

The `InventoryAnalysis` class in `inventory_analysis.py` performs an in-depth analysis of the simulation results. It calculates key performance metrics such as average inventory levels, stockout rates, and service levels.

#### Key Functions:
- **`analyze`**: Analyzes the simulation results and calculates performance metrics.
- **`format_report`**: Formats the analysis results into a readable report.

### Inventory Policies

The `inventory_policies.py` module implements various inventory replenishment policies. These policies determine when and how much to reorder based on the current inventory levels, demand, and lead times.

#### Key Functions:
1. **Predict-Then-Optimize (PTO) Policy**:
##### **Description:**
- Dynamic safety stock-based ordering
- Uses demand forecasting (`future_demand`) and lead time estimation to calculate required orders
- Maintains safety stock buffer to meet service level requirements

##### **Ordering Condition:**
Orders when projected inventory dips below:
```
demand_forecast × lead_time + safety_stock
demand_forecast × interval + safety_stock
```

##### **Features:**
- Demand-responsive ordering
- Probabilistic safety buffers

---

2. **Fixed Order Quantity Policy**
##### **Description:**
- Constant batch ordering
- Orders predefined quantity (`fixed_reorder_quantity`) when triggered
- Ignores current inventory level except for reorder point timing

##### **Features:**
- Simple implementation
- Consistent order sizes
- Minimal calculations required

--- 

3. **Order-Up-To Policy**
#####  **Description:**
- Target inventory level maintenance
- Replenishes inventory to fixed `order_up_to_point`

**Order Quantity Calculation:**
```
max(0, target_level - current_inventory)
```

##### **Features:**
- Gap-filling approach
- Maintains constant maximum inventory
- Demand-agnostic ordering logic

---

4. **E2E (End-to-End) Policy**
##### **Description:**
- Cost-optimized time-phased ordering
- Balances holding costs (`h`) and backorder costs (`b`)
- Uses planned order schedule (`po_reaching_e2e`) to:
  - Calculate optimal order timing (`delta_t`)
  - Determine quantity to cover demand until next replenishment

---

## Usage

To run the project, follow these steps:

1. **Clone the Repository**: \
Clone the repository or download the script to your local machine.

2. **Install Dependencies**: \
Ensure you have Python installed, and then install the required libraries:
```bash
pip install numpy pandas matplotlib 
```

3. **Run the Simulation**:
Execute the `main.py` script to run the simulation, visualize the results, and generate the analysis report:
```bash
python main.py
```

### Example Output
The script will simulate inventory management under different demand distributions (Normal, Gamma, Lognormal, Poisson) and generate the following outputs:
- **Plots**: Visualizations of inventory levels and purchase orders over time.
- **CSV Files**: Exported simulation data for further analysis.
- **Analysis Report**: A detailed report comparing the performance of different inventory policies.
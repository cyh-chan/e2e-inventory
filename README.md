# End-to-End Inventory Management Repository

This repository is a work-in-progress extension of the project *"A Practical End-to-End Inventory Management Model with Deep Learning"*. It provides a comprehensive framework for simulating, analyzing, and optimizing inventory management systems using deep learning and traditional inventory policies. The repository is organized into multiple folders, each focusing on a specific aspect of inventory management.

## Table of Contents
1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Folder Descriptions](#folder-descriptions)
   - [e2e_data_generation](#e2e_data_generation)
   - [e2e_pytorch](#e2e_pytorch)
   - [e2e_tf](#e2e_tf)
   - [policies_simulation](#policies_simulation)
4. [Getting Started](#getting-started)
---

## Overview

This repository is designed to explore and implement end-to-end inventory management solutions using both deep learning models (PyTorch and TensorFlow) and traditional inventory policies. The goal is to provide a practical framework for simulating inventory systems, generating synthetic data, training deep learning models, and evaluating the performance of different inventory management strategies.

The repository is structured into four main folders, each addressing a specific component of the inventory management pipeline. For detailed documentation, please refer to the respective `README.md` files in each folder.

---

## Repository Structure

The repository is organized as follows: \
e2e-inventory/ \
├── e2e_data_generation/ # Synthetic data generation for inventory systems \
├── e2e_pytorch/ # PyTorch-based deep learning models for inventory management \
├── e2e_tf/ # TensorFlow-based deep learning models for inventory management \
├── policies_simulation/ # Simulation of traditional inventory policies \
└── README.md # This file 


---

## Folder Descriptions

### e2e_data_generation

This folder contains scripts and utilities for generating synthetic inventory data. The data includes features such as demand, lead time, inventory levels, and reorder points, which are essential for training and evaluating inventory management models.

- **Key Features**:
  - Generate synthetic demand and lead time data using various statistical distributions.
  - Simulate inventory levels and reorder points based on predefined policies.
  - Export data in CSV and Parquet formats for further analysis.

---

### e2e_pytorch

This folder implements end-to-end inventory management models using **PyTorch**. The models are designed to predict optimal reorder quantities, forecast demand, and optimize inventory levels using deep learning techniques.

- **Key Features**:
  - Dilated convolutional neural networks for sequential data processing.
  - Custom loss functions tailored for inventory management tasks.
  - Visualization and analysis tools for model evaluation.

---

### e2e_tf

This folder implements end-to-end inventory management models using **TensorFlow**. Similar to the PyTorch implementation, these models focus on demand forecasting, reorder optimization, and inventory level management.

- **Key Features**:
  - TensorFlow-based neural networks for inventory prediction.
  - Integration with TensorBoard for training visualization.
  - Support for exporting trained models for deployment.

---

### policies_simulation

This folder simulates traditional inventory management policies, such as Predict-Then-Optimize (PTO) policy. It provides a baseline for comparing the performance of deep learning models against classical approaches.

- **Key Features**:
  - Simulation of multiple inventory policies under various demand distributions.
  - Visualization tools for comparing policy performance.
  - Detailed analysis of key metrics such as stockout rates and holding costs.

---

## Getting Started

To get started with this repository, follow these steps:

1. **Clone the Repository**:

2. **Set Up the Environment**:
Install the required dependencies for each folder. Refer to the respective README.md files for detailed instructions.

3. **Explore the Folders**:
Navigate to the folder of interest (i.e. e2e_data_generation, e2e_pytorch, etc.) and follow the instructions provided in the README.md file to run the scripts and experiments.

4. **Run Simulations and Train Models**:
Use the provided scripts to generate data, train models, and simulate inventory policies. Compare the results to gain insights into the performance of different approaches.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

def visualize_grouped_data(data_grouped):
    """
    Create multiple visualizations for each SKU group in the inventory data
    """
    for sku, group in data_grouped:
        plt.figure(figsize=(20, 15))

        # 1. Distribution of Demand
        plt.subplot(2, 3, 1)
        plt.hist(group['demand'], bins=30, edgecolor='black')
        plt.title(f'Distribution of Demand for SKU {sku}')
        plt.xlabel('Demand')
        plt.ylabel('Frequency')

        # 2. Distribution of Lead Time
        plt.subplot(2, 3, 2)
        plt.hist(group['lead_time'], bins=20, edgecolor='black')
        plt.title(f'Distribution of Lead Time for SKU {sku}')
        plt.xlabel('Lead Time')
        plt.ylabel('Frequency')

        # 3. Boxplot of Demand
        plt.subplot(2, 3, 3)
        plt.boxplot(group['demand'])
        plt.title(f'Boxplot of Demand for SKU {sku}')
        plt.ylabel('Demand')

        # 4. Reorder Point Distribution
        plt.subplot(2, 3, 4)
        plt.hist(group['reorder_point'], bins=20, edgecolor='black')
        plt.title(f'Distribution of Reorder Points for SKU {sku}')
        plt.xlabel('Reorder Point')
        plt.ylabel('Frequency')

        # 5. Initial Stock Level Distribution
        plt.subplot(2, 3, 5)
        plt.hist(group['inventory'], bins=30, edgecolor='black')
        plt.title(f'Distribution of Initial Stock Levels for SKU {sku}')
        plt.xlabel('Initial Stock')
        plt.ylabel('Frequency')

        # 6. Bar Plot of Mean Values
        plt.subplot(2, 3, 6)
        means = [
            np.mean(group['demand']),
            np.mean(group['lead_time']),
            np.mean(group['reorder_point']),
            np.mean(group['inventory'])
        ]
        plt.bar(['Demand', 'Lead Time', 'Reorder Point', 'Initial Stock'], means)
        plt.title(f'Mean Values for SKU {sku}')
        plt.xticks(rotation=45)
        plt.ylabel('Mean')

        plt.tight_layout()
        plt.show()

        # Detailed statistics
        print(f"Data Diagnostics for SKU {sku}:")
        for key, data in [
            ('Demand', group['demand']),
            ('Lead Time', group['lead_time']),
            ('Reorder Point', group['reorder_point']),
            ('Initial Stock', group['inventory'])
        ]:
            print(f"\n{key} Statistics:")
            print(f"Mean: {np.mean(data):.2f}")
            print(f"Std Dev: {np.std(data):.2f}")
            print(f"Min: {np.min(data)}")
            print(f"Max: {np.max(data)}")


def summarize_dataframe(data):
    """
    Output a summary of the entire DataFrame, including statistics for numerical columns
    and counts for categorical columns.
    """
    print("DataFrame Summary:")
    print("==================")

    # General Information
    print(f"\nShape of DataFrame: {data.shape}")
    print(f"Total Rows: {data.shape[0]}")
    print(f"Total Columns: {data.shape[1]}")
    print("\nColumn Names:", data.columns.tolist())

    # Data Types
    print("\nData Types:")
    print(data.dtypes)

    # Missing Values
    print("\nMissing Values:")
    print(data.isnull().sum())

    # Numerical Columns Summary
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        print("\nSummary for Numerical Columns:")
        print(data[numerical_cols].describe().transpose())

    # Categorical Columns Summary
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print("\nSummary for Categorical Columns:")
        for col in categorical_cols:
            print(f"\nColumn: {col}")
            print(f"Unique Values: {data[col].nunique()}")
            print(f"Top Value: {data[col].mode().values[0]}")
            print(f"Frequency of Top Value: {data[col].value_counts().iloc[0]}")

    # Correlation Matrix for Numerical Columns
    if len(numerical_cols) > 1:
        print("\nCorrelation Matrix for Numerical Columns:")
        print(data[numerical_cols].corr())

    print("\nEnd of Summary.")

def main():
    # Read data from Parquet file
    data = pd.read_parquet('inventory_dataset.parquet')

    # Group data by SKU
    data_grouped = data.groupby('sku')

    # Visualize the data for each SKU
    # visualize_grouped_data(data_grouped)

    # Output summary for the entire DataFrame
    summarize_dataframe(data)

    # ydata profiling
    # Generate profile report
    profile = ProfileReport(data, explorative=True)
    profile.to_file("inventory_report.html")
    print("HTML report saved as inventory_report.html")

    # Print first few rows to verify
    print(data.head())

if __name__ == "__main__":
    main()
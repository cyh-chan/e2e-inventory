# inventory_analysis.py

import numpy as np
import scipy.stats as stats
import statsmodels.api as sm


def analyze_inventory_performance(inventory_levels, demand, holding_cost, backorder_cost, time_periods=None):
    # Initialize results dictionary
    results = {}

    # Calculate holding and stockout quantities
    positive_inventory = np.maximum(inventory_levels, 0)
    stockouts = np.maximum(-inventory_levels, 0)

    # Calculate costs
    holding_costs = holding_cost * positive_inventory
    stockout_costs = backorder_cost * stockouts
    total_costs = holding_costs + stockout_costs

    # Calculate averages
    results['avg_holding_cost'] = np.mean(holding_costs)
    results['avg_stockout_cost'] = np.mean(stockout_costs)
    results['avg_total_cost'] = np.mean(total_costs)

    # Calculate turnover rate (annual)
    total_demand = np.sum(demand)
    avg_inventory = np.mean(positive_inventory)
    results['turnover_rate'] = (total_demand / avg_inventory) * (365 / len(inventory_levels))

    # Calculate stockout ratio
    stockout_periods = np.sum(stockouts > 0)
    results['stockout_ratio'] = stockout_periods / len(inventory_levels)

    # Perform t-tests
    # H0: mean cost = 0 vs H1: mean cost ≠ 0
    results['t_tests'] = {}

    for cost_type, costs in [
        ('holding_cost', holding_costs),
        ('stockout_cost', stockout_costs),
        ('total_cost', total_costs)
    ]:
        t_stat, p_value = stats.ttest_1samp(costs, 0)
        results['t_tests'][cost_type] = {
            't_statistic': t_stat,
            'p_value': p_value
        }

    # Perform OLS regression
    # Create time index
    time_index = np.arange(len(inventory_levels))

    # Prepare data for regression
    X = sm.add_constant(np.column_stack((
        time_index,
        demand,
        positive_inventory
    )))

    # Run regression for each cost type
    results['regression'] = {}

    for cost_type, y in [
        ('holding_cost', holding_costs),
        ('stockout_cost', stockout_costs),
        ('total_cost', total_costs)
    ]:
        model = sm.OLS(y, X)
        regression = model.fit()
        results['regression'][cost_type] = {
            'params': regression.params,
            'r_squared': regression.rsquared,
            'adj_r_squared': regression.rsquared_adj,
            'f_stat': regression.fvalue,
            'f_pvalue': regression.f_pvalue,
            'summary': regression.summary()
        }

    # If time periods provided, calculate comparative statistics
    if time_periods:
        train_start, train_end, test_start, test_end = time_periods
        results['comparative'] = {
            'train': {
                'avg_holding_cost': np.mean(holding_costs[train_start:train_end]),
                'avg_stockout_cost': np.mean(stockout_costs[train_start:train_end]),
                'avg_total_cost': np.mean(total_costs[train_start:train_end]),
                'turnover_rate': (np.sum(demand[train_start:train_end]) /
                                  np.mean(positive_inventory[train_start:train_end])) *
                                 (365 / (train_end - train_start)),
                'stockout_ratio': np.sum(stockouts[train_start:train_end] > 0) /
                                  (train_end - train_start)
            },
            'test': {
                'avg_holding_cost': np.mean(holding_costs[test_start:test_end]),
                'avg_stockout_cost': np.mean(stockout_costs[test_start:test_end]),
                'avg_total_cost': np.mean(total_costs[test_start:test_end]),
                'turnover_rate': (np.sum(demand[test_start:test_end]) /
                                  np.mean(positive_inventory[test_start:test_end])) *
                                 (365 / (test_end - test_start)),
                'stockout_ratio': np.sum(stockouts[test_start:test_end] > 0) /
                                  (test_end - test_start)
            }
        }

    return results


def print_analysis_results(results):
    print("\n=== Inventory Performance Analysis ===\n")

    print("Cost Metrics:")
    print(f"Average Holding Cost: {results['avg_holding_cost']:.2f}")
    print(f"Average Stockout Cost: {results['avg_stockout_cost']:.2f}")
    print(f"Average Total Cost: {results['avg_total_cost']:.2f}")

    print("\nPerformance Metrics:")
    print(f"Inventory Turnover Rate (annual): {results['turnover_rate']:.2f}")
    print(f"Stockout Ratio: {results['stockout_ratio']:.2%}")

    print("\nStatistical Tests:")
    for cost_type, test_results in results['t_tests'].items():
        print(f"\n{cost_type.replace('_', ' ').title()} T-Test:")
        print(f"t-statistic: {test_results['t_statistic']:.4f}")
        print(f"p-value: {test_results['p_value']:.4f}")

    print("\nRegression Analysis:")
    for cost_type, reg_results in results['regression'].items():
        print(f"\n{cost_type.replace('_', ' ').title()} Regression:")
        print(f"R-squared: {reg_results['r_squared']:.4f}")
        print(f"Adjusted R-squared: {reg_results['adj_r_squared']:.4f}")
        print(f"F-statistic: {reg_results['f_stat']:.4f}")
        print(f"F-statistic p-value: {reg_results['f_pvalue']:.4f}")

    if 'comparative' in results:
        print("\nComparative Analysis:")
        metrics = ['avg_holding_cost', 'avg_stockout_cost', 'avg_total_cost',
                   'turnover_rate', 'stockout_ratio']

        print("\nMetric               Training    Testing     % Change")
        print("-" * 50)
        for metric in metrics:
            train_val = results['comparative']['train'][metric]
            test_val = results['comparative']['test'][metric]
            pct_change = ((test_val - train_val) / train_val) * 100

            if metric == 'stockout_ratio':
                print(f"{metric.replace('_', ' ').title():20} {train_val:.2%}    {test_val:.2%}    {pct_change:+.1f}%")
            else:
                print(f"{metric.replace('_', ' ').title():20} {train_val:.2f}    {test_val:.2f}    {pct_change:+.1f}%")
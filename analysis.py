import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

def analyze_inventory_policies(
        inventory_level_ss,
        inventory_level_sq,
        inventory_level_labelling,
        inventory_level_out,
        future_demand,
        h=1,  # holding cost per unit
        b=9,  # stockout cost per unit
        confidence_level=0.95
):
    """
    Comprehensive analysis of different inventory policies.

    Parameters:
    - inventory_level_*: Arrays of inventory levels for each policy
    - future_demand: Array of demand values
    - h: Holding cost per unit
    - b: Stockout cost per unit
    - confidence_level: Confidence level for statistical tests

    Returns:
    - Dictionary containing all analysis results
    """
    import numpy as np
    from scipy import stats
    from sklearn.linear_model import LinearRegression

    policies = {
        'SS Policy': inventory_level_ss,
        'Fixed Order-Quantity': inventory_level_sq,
        'Labelling Policy': inventory_level_labelling,
        'Order-Up-To': inventory_level_out
    }

    results = {
        'cost_metrics': {},
        'performance_metrics': {},
        'statistical_tests': {},
        'regression_analysis': {},
        'comparative_analysis': {},
        'diff_in_diff': {}
    }

    # 1. Cost Metrics Analysis
    for name, inventory in policies.items():
        holding_cost = np.sum(np.maximum(inventory, 0) * h)
        stockout_cost = np.sum(np.maximum(-inventory, 0) * b)
        total_cost = holding_cost + stockout_cost

        results['cost_metrics'][name] = {
            'holding_cost': holding_cost,
            'stockout_cost': stockout_cost,
            'total_cost': total_cost
        }

    # 2. Performance Metrics
    for name, inventory in policies.items():
        # Inventory Turnover Rate
        avg_inventory = np.mean(np.maximum(inventory, 0))
        total_demand = np.sum(future_demand)
        turnover_rate = total_demand / avg_inventory if avg_inventory > 0 else 0

        # Stockout Rate
        stockout_periods = np.sum(inventory < 0)
        stockout_rate = stockout_periods / len(inventory)

        results['performance_metrics'][name] = {
            'turnover_rate': turnover_rate,
            'stockout_rate': stockout_rate,
            'average_inventory': avg_inventory
        }

    # 3. Statistical Tests
    # Perform paired t-tests between policies
    policy_names = list(policies.keys())
    for i in range(len(policy_names)):
        for j in range(i + 1, len(policy_names)):
            policy1 = policy_names[i]
            policy2 = policy_names[j]

            t_stat, p_value = stats.ttest_ind(
                policies[policy1],
                policies[policy2]
            )

            results['statistical_tests'][f'{policy1} vs {policy2}'] = {
                't_statistic': t_stat,
                'p_value': p_value
            }

    # 4. Regression Analysis
    for name, inventory in policies.items():
        X = np.arange(len(inventory)).reshape(-1, 1)
        y = inventory

        model = LinearRegression()
        model.fit(X, y)

        r2 = model.score(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_

        results['regression_analysis'][name] = {
            'r2': r2,
            'slope': slope,
            'intercept': intercept
        }

    # 5. Comparative Analysis
    baseline = np.mean(inventory_level_labelling)  # Using SS Policy as baseline
    for name, inventory in policies.items():
        if name != 'Labelling Policy':
            relative_performance = (np.mean(inventory) - baseline) / baseline
            results['comparative_analysis'][name] = {
                'relative_to_baseline': relative_performance
            }

    # 6. Difference-in-Difference Analysis
    # Compare pre/post period effects between policies
    mid_point = len(inventory_level_ss) // 2

    for name, inventory in policies.items():
        if name != 'Labelling Policy':
            # Pre-period difference
            pre_diff = np.mean(inventory[:mid_point]) - np.mean(inventory_level_labelling[:mid_point])
            # Post-period difference
            post_diff = np.mean(inventory[mid_point:]) - np.mean(inventory_level_labelling[mid_point:])
            # DiD estimate
            did_estimate = post_diff - pre_diff

            results['diff_in_diff'][name] = {
                'pre_difference': pre_diff,
                'post_difference': post_diff,
                'did_estimate': did_estimate
            }

    return results


def format_analysis_report(results):
    report = "Inventory Policy Analysis Report\n"
    report += "=" * 30 + "\n\n"

    # Cost Metrics
    report += "1. Cost Metrics Analysis\n"
    report += "-" * 20 + "\n"
    for policy, metrics in results['cost_metrics'].items():
        report += f"\n{policy}:\n"
        report += f"  Holding Cost: {metrics['holding_cost']:.2f}\n"
        report += f"  Stockout Cost: {metrics['stockout_cost']:.2f}\n"
        report += f"  Total Cost: {metrics['total_cost']:.2f}\n"

    # Performance Metrics
    report += "\n2. Performance Metrics\n"
    report += "-" * 20 + "\n"
    for policy, metrics in results['performance_metrics'].items():
        report += f"\n{policy}:\n"
        report += f"  Turnover Rate: {metrics['turnover_rate']:.2f}\n"
        report += f"  Stockout Rate: {metrics['stockout_rate']:.2%}\n"
        report += f"  Average Inventory: {metrics['average_inventory']:.2f}\n"

    # Statistical Tests
    report += "\n3. Statistical Tests\n"
    report += "-" * 20 + "\n"
    for comparison, stats_results in results['statistical_tests'].items():
        report += f"\n{comparison}:\n"
        report += f"  t-statistic: {stats_results['t_statistic']:.3f}\n"
        report += f"  p-value: {stats_results['p_value']:.3f}\n"

    # Regression Analysis
    report += "\n4. Regression Analysis\n"
    report += "-" * 20 + "\n"
    for policy, reg_results in results['regression_analysis'].items():
        report += f"\n{policy}:\n"
        report += f"  R² Score: {reg_results['r2']:.3f}\n"
        report += f"  Slope: {reg_results['slope']:.3f}\n"
        report += f"  Intercept: {reg_results['intercept']:.3f}\n"

    # Comparative Analysis
    report += "\n5. Comparative Analysis (Relative to Labelling Policy)\n"
    report += "-" * 20 + "\n"
    for policy, comp_results in results['comparative_analysis'].items():
        report += f"\n{policy}:\n"
        report += f"  Relative Performance: {comp_results['relative_to_baseline']:.2%}\n"

    # Difference-in-Difference
    report += "\n6. Difference-in-Difference Analysis\n"
    report += "-" * 20 + "\n"
    for policy, did_results in results['diff_in_diff'].items():
        report += f"\n{policy}:\n"
        report += f"  Pre-period Difference: {did_results['pre_difference']:.2f}\n"
        report += f"  Post-period Difference: {did_results['post_difference']:.2f}\n"
        report += f"  DiD Estimate: {did_results['did_estimate']:.2f}\n"

    return report

import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

class InventoryAnalysis:
    def __init__(self, inventory_level_pto, inventory_level_sq, inventory_level_e2e, inventory_level_out, future_demand, h=1, b=9, confidence_level=0.95):
        self.inventory_level_pto = inventory_level_pto
        self.inventory_level_sq = inventory_level_sq
        self.inventory_level_e2e = inventory_level_e2e
        self.inventory_level_out = inventory_level_out
        self.future_demand = future_demand
        self.h = h
        self.b = b
        self.confidence_level = confidence_level

    def analyze(self):
        policies = {
            'PTO Policy': self.inventory_level_pto,
            'Fixed Order-Quantity': self.inventory_level_sq,
            'E2E Policy': self.inventory_level_e2e,
            'Order-Up-To': self.inventory_level_out
        }

        results = {
            'cost_metrics': {},
            'performance_metrics': {},
            'statistical_tests': {},
            'regression_analysis': {},
            'comparative_analysis': {},
            'diff_in_diff': {}
        }

        for name, inventory in policies.items():
            holding_cost = np.sum(np.maximum(inventory, 0) * self.h)
            stockout_cost = np.sum(np.maximum(-inventory, 0) * self.b)
            total_cost = holding_cost + stockout_cost

            results['cost_metrics'][name] = {
                'holding_cost': holding_cost,
                'stockout_cost': stockout_cost,
                'total_cost': total_cost
            }

        for name, inventory in policies.items():
            avg_inventory = np.mean(np.maximum(inventory, 0))
            total_demand = np.sum(self.future_demand)
            turnover_rate = total_demand / avg_inventory if avg_inventory > 0 else 0

            stockout_periods = np.sum(inventory < 0)
            stockout_rate = stockout_periods / len(inventory)

            results['performance_metrics'][name] = {
                'turnover_rate': turnover_rate,
                'stockout_rate': stockout_rate,
                'average_inventory': avg_inventory
            }

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

        baseline = np.mean(self.inventory_level_e2e)
        for name, inventory in policies.items():
            if name != 'E2E Policy':
                relative_performance = (np.mean(inventory) - baseline) / baseline
                results['comparative_analysis'][name] = {
                    'relative_to_baseline': relative_performance
                }

        mid_point = len(self.inventory_level_pto) // 2

        for name, inventory in policies.items():
            if name != 'E2E Policy':
                pre_diff = np.mean(inventory[:mid_point]) - np.mean(self.inventory_level_e2e[:mid_point])
                post_diff = np.mean(inventory[mid_point:]) - np.mean(self.inventory_level_e2e[mid_point:])
                did_estimate = post_diff - pre_diff

                results['diff_in_diff'][name] = {
                    'pre_difference': pre_diff,
                    'post_difference': post_diff,
                    'did_estimate': did_estimate
                }

        return results

    def format_report(self, results):
        report = "Inventory Policy Analysis Report\n"
        report += "=" * 30 + "\n\n"

        report += "1. Cost Metrics Analysis\n"
        report += "-" * 20 + "\n"
        for policy, metrics in results['cost_metrics'].items():
            report += f"\n{policy}:\n"
            report += f"  Holding Cost: {metrics['holding_cost']:.2f}\n"
            report += f"  Stockout Cost: {metrics['stockout_cost']:.2f}\n"
            report += f"  Total Cost: {metrics['total_cost']:.2f}\n"

        report += "\n2. Performance Metrics\n"
        report += "-" * 20 + "\n"
        for policy, metrics in results['performance_metrics'].items():
            report += f"\n{policy}:\n"
            report += f"  Turnover Rate: {metrics['turnover_rate']:.2f}\n"
            report += f"  Stockout Rate: {metrics['stockout_rate']:.2%}\n"
            report += f"  Average Inventory: {metrics['average_inventory']:.2f}\n"
        report += "\n3. Statistical Tests\n"
        report += "-" * 20 + "\n"
        for comparison, stats_results in results['statistical_tests'].items():
            report += f"\n{comparison}:\n"
            report += f"  t-statistic: {stats_results['t_statistic']:.3f}\n"
            report += f"  p-value: {stats_results['p_value']:.3f}\n"
        report += "\n4. Regression Analysis\n"
        report += "-" * 20 + "\n"
        for policy, reg_results in results['regression_analysis'].items():
            report += f"\n{policy}:\n"
            report += f"  R² Score: {reg_results['r2']:.3f}\n"
            report += f"  Slope: {reg_results['slope']:.3f}\n"
            report += f"  Intercept: {reg_results['intercept']:.3f}\n"

        report += "\n5. Comparative Analysis (Relative to E2E Policy)\n"
        report += "-" * 20 + "\n"
        for policy, comp_results in results['comparative_analysis'].items():
            report += f"\n{policy}:\n"
            report += f"  Relative Performance: {comp_results['relative_to_baseline']:.2%}\n"

        report += "\n6. Difference-in-Difference Analysis\n"
        report += "-" * 20 + "\n"
        for policy, did_results in results['diff_in_diff'].items():
            report += f"\n{policy}:\n"
            report += f"  Pre-period Difference: {did_results['pre_difference']:.2f}\n"
            report += f"  Post-period Difference: {did_results['post_difference']:.2f}\n"
            report += f"  DiD Estimate: {did_results['did_estimate']:.2f}\n"

        return report
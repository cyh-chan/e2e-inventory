import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import csv
import scipy.stats as stats
from invutils import get_safety_stock
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def set_random_seed(seed=42):
    np.random.seed(seed)

class InventorySimulation:
    def __init__(self, initial_inventory=100, horizon=60, reorder_days=[4], service_level=0.99,
                 demand_distribution='normal', demand_params=None, lead_time_distribution='normal',
                 lead_time_params=None, b=9, h=1, fixed_reorder_quantity=100, order_up_to_point=200):
        self.initial_inventory = initial_inventory
        self.horizon = horizon
        self.reorder_days = reorder_days
        self.service_level = service_level
        self.demand_distribution = demand_distribution
        self.demand_params = demand_params if demand_params else {'mu': 10, 'std': 6}
        self.lead_time_distribution = lead_time_distribution
        self.lead_time_params = lead_time_params if lead_time_params else {'mu': 5, 'std': 1}
        self.b = b
        self.h = h
        self.fixed_reorder_quantity = fixed_reorder_quantity
        self.order_up_to_point = order_up_to_point

    def generate_demand(self, distribution, params, size):
        distributions = {
            'normal': lambda: stats.truncnorm.rvs(
                (0 - params.get('mu', 10)) / params.get('std', 6),
                np.inf,  # No upper bound
                loc=params.get('mu', 10),
                scale=params.get('std', 6),
                size=size
            ),
            'gamma': lambda: stats.gamma.rvs(a=params.get('shape', 2),
                                             scale=params.get('scale', 1),
                                             size=size),
            'lognormal': lambda: stats.lognorm.rvs(s=params.get('sigma', 0.5),
                                                   scale=np.exp(params.get('mu', 2)),
                                                   size=size),
            'poisson': lambda: stats.poisson.rvs(mu=max(params.get('lambda', 2), 0.1),
                                                 size=size)  # Ensure lambda > 0
        }

        if distribution not in distributions:
            raise ValueError(f"Unsupported distribution: {distribution}")

        return distributions[distribution]()

    def generate_lead_time(self, distribution, params, size):
        return np.abs(self.generate_demand(distribution, params, size)).astype(int)

    def create_reorder_point_array(self, horizon, reorder_days):
        start_date = datetime(2025, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(horizon)]
        reorder_point_arr = np.zeros(horizon, dtype=bool)
        for i, date in enumerate(dates):
            reorder_point_arr[i] = date.weekday() in reorder_days
        return reorder_point_arr

    def run_simulation(self):
        future_demand = self.generate_demand(self.demand_distribution, self.demand_params, self.horizon)
        reorder_point_arr = self.create_reorder_point_array(self.horizon, self.reorder_days)
        n_reorder_pts = reorder_point_arr.sum()
        future_lead_time = self.generate_lead_time(self.lead_time_distribution, self.lead_time_params, n_reorder_pts)
        history_lead_time = self.generate_lead_time(self.lead_time_distribution, self.lead_time_params, 50)
        history_demand = self.generate_demand(self.demand_distribution, self.demand_params, 50)

        safety_stock = self.get_safety_stock(history_lead_time, history_demand, self.service_level)

        ss_policy = SafetyStockPolicy(self.initial_inventory, safety_stock, self.service_level)
        fixed_policy = FixedOrderQuantityPolicy(self.initial_inventory, self.fixed_reorder_quantity)
        upto_policy = OrderUpToPolicy(self.initial_inventory, self.order_up_to_point)
        po_reaching_labelling = list(np.where(reorder_point_arr)[0] + np.ceil(future_lead_time).astype(int))
        po_reaching_labelling.append(po_reaching_labelling[-1] + int(np.diff(po_reaching_labelling).mean()))
        labelling_policy = LabellingPolicy(self.initial_inventory, self.b, self.h)

        policies = {
            'ss': ss_policy,
            'fixed': fixed_policy,
            'upto': upto_policy,
            'labelling': labelling_policy
        }

        for i in range(1, self.horizon):
            for name, policy in policies.items():
                policy.receive_shipment(i)
                policy.update_inventory(i, future_demand[i])
                if name == 'labelling':
                    reorder_qty = policy.reorder(i, reorder_point_arr[i], future_demand, future_lead_time, self.horizon, po_reaching_labelling)
                else:
                    reorder_qty = policy.reorder(i, reorder_point_arr[i], future_demand, future_lead_time, self.horizon)

        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(self.horizon)]
        results = {
            'dates': dates,
            'future_demand': future_demand,
            'inventory_level_ss': ss_policy.inventory_level,
            'inventory_level_sq': fixed_policy.inventory_level,
            'inventory_level_out': upto_policy.inventory_level,
            'inventory_level_labelling': labelling_policy.inventory_level,
            'po_ss': ss_policy.po,
            'po_sq': fixed_policy.po,
            'po_out': upto_policy.po,
            'po_labelling': labelling_policy.po,
            'po_reaching_ss': ss_policy.po_reaching,
            'po_reaching_sq': fixed_policy.po_reaching,
            'po_reaching_out': upto_policy.po_reaching,
            'po_reaching_labelling': labelling_policy.actual_po_reaching,
            'po_reaching_labelling_schedule': po_reaching_labelling
        }

        return results

    def get_safety_stock(self, history_lead_time, history_demand, service_level):
        # Placeholder for safety stock calculation
        return np.mean(history_demand) * np.mean(history_lead_time) * service_level

class InventoryPolicy:
    def __init__(self, initial_inventory):
        self.inventory_level = np.zeros(60)
        self.inventory_level[0] = initial_inventory
        self.po = {}
        self.po_reaching = []
        self.pending_order = None
        self.reorder_ptr = 0

    def reorder(self, day, reorder_point, future_demand, future_lead_time, horizon):
        raise NotImplementedError("Subclasses must implement the reorder method.")

    def update_inventory(self, day, demand):
        self.inventory_level[day] = self.inventory_level[day-1] + self.inventory_level[day] - demand

    def receive_shipment(self, day):
        if self.pending_order and day >= self.pending_order['delivery_date']:
            self.inventory_level[day] += self.pending_order['quantity']
            self.po_reaching.append(self.pending_order['delivery_date'])
            self.pending_order = None

class SafetyStockPolicy(InventoryPolicy):
    def __init__(self, initial_inventory, safety_stock, service_level):
        super().__init__(initial_inventory)
        self.safety_stock = safety_stock
        self.service_level = service_level

    def reorder(self, day, reorder_point, future_demand, future_lead_time, horizon):
        if self.pending_order:
            return 0

        if not reorder_point:
            return 0

        interval = 7
        demand_forecast = abs(future_demand[self.reorder_ptr] + np.random.normal(0, 0))
        reorder_qty = 0
        delta1 = self.inventory_level[day] - demand_forecast * future_lead_time[self.reorder_ptr] - self.safety_stock
        if delta1 < 0:
            reorder_qty = -delta1

        delta2 = self.inventory_level[day] - demand_forecast * interval - self.safety_stock
        if delta2 < 0:
            reorder_qty = max(reorder_qty, -delta2)

        if reorder_qty > 0:
            delivery_date = day + int(np.ceil(future_lead_time[self.reorder_ptr]))
            if delivery_date < horizon:
                self.pending_order = {
                    'quantity': reorder_qty,
                    'delivery_date': delivery_date
                }
                self.po[day] = reorder_qty
                self.reorder_ptr += 1

        return reorder_qty

class FixedOrderQuantityPolicy(InventoryPolicy):
    def __init__(self, initial_inventory, fixed_reorder_quantity):
        super().__init__(initial_inventory)
        self.fixed_reorder_quantity = fixed_reorder_quantity

    def reorder(self, day, reorder_point, future_demand, future_lead_time, horizon):
        if self.pending_order:
            return 0

        if not reorder_point:
            return 0

        reorder_qty = self.fixed_reorder_quantity
        delivery_date = day + int(np.ceil(future_lead_time[self.reorder_ptr]))
        if delivery_date < horizon:
            self.pending_order = {
                'quantity': reorder_qty,
                'delivery_date': delivery_date
            }
            self.po[day] = reorder_qty
            self.reorder_ptr += 1

        return reorder_qty

class OrderUpToPolicy(InventoryPolicy):
    def __init__(self, initial_inventory, order_up_to_point):
        super().__init__(initial_inventory)
        self.order_up_to_point = order_up_to_point

    def reorder(self, day, reorder_point, future_demand, future_lead_time, horizon):
        if self.pending_order:
            return 0

        if not reorder_point:
            return 0

        inventory_before_demand = self.inventory_level[day - 1]
        reorder_qty = max(0, self.order_up_to_point - inventory_before_demand)

        if reorder_qty > 0:
            delivery_date = day + int(np.ceil(future_lead_time[self.reorder_ptr]))
            if delivery_date < horizon:
                self.pending_order = {
                    'quantity': reorder_qty,
                    'delivery_date': delivery_date
                }
                self.po[day] = reorder_qty
                self.reorder_ptr += 1

        return reorder_qty

class LabellingPolicy(InventoryPolicy):
    def __init__(self, initial_inventory, b, h):
        super().__init__(initial_inventory)
        self.b = b
        self.h = h
        self.t = 0
        self.actual_po_reaching = []

    def reorder(self, day, reorder_point, future_demand, future_lead_time, horizon, po_reaching_labelling):
        if self.pending_order:
            return 0

        if not reorder_point:
            return 0

        if self.t >= len(po_reaching_labelling):
            return 0

        vm = po_reaching_labelling[self.t]
        if vm >= horizon:
            return 0

        delta_t = int(np.floor(self.b * (po_reaching_labelling[self.t + 1] - po_reaching_labelling[self.t]) / (self.h + self.b)))
        delta_t = min(delta_t, horizon - vm)
        cum_demand = np.sum(future_demand[vm:min(vm + delta_t, horizon)])
        optimal_qty = max(cum_demand + np.sum(future_demand[day:vm]) - self.inventory_level[day], 0)

        if optimal_qty > 0:
            delivery_date = vm
            if delivery_date < horizon:
                self.pending_order = {
                    'quantity': optimal_qty,
                    'delivery_date': delivery_date
                }
                self.po[day] = optimal_qty
                self.actual_po_reaching.append(vm)
                self.t += 1

        return optimal_qty

class InventoryAnalysis:
    def __init__(self, inventory_level_ss, inventory_level_sq, inventory_level_labelling, inventory_level_out, future_demand, h=1, b=9, confidence_level=0.95):
        self.inventory_level_ss = inventory_level_ss
        self.inventory_level_sq = inventory_level_sq
        self.inventory_level_labelling = inventory_level_labelling
        self.inventory_level_out = inventory_level_out
        self.future_demand = future_demand
        self.h = h
        self.b = b
        self.confidence_level = confidence_level

    def analyze(self):
        policies = {
            'SS Policy': self.inventory_level_ss,
            'Fixed Order-Quantity': self.inventory_level_sq,
            'Labelling Policy': self.inventory_level_labelling,
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

        baseline = np.mean(self.inventory_level_labelling)
        for name, inventory in policies.items():
            if name != 'Labelling Policy':
                relative_performance = (np.mean(inventory) - baseline) / baseline
                results['comparative_analysis'][name] = {
                    'relative_to_baseline': relative_performance
                }

        mid_point = len(self.inventory_level_ss) // 2

        for name, inventory in policies.items():
            if name != 'Labelling Policy':
                pre_diff = np.mean(inventory[:mid_point]) - np.mean(self.inventory_level_labelling[:mid_point])
                post_diff = np.mean(inventory[mid_point:]) - np.mean(self.inventory_level_labelling[mid_point:])
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

        report += "\n5. Comparative Analysis (Relative to Labelling Policy)\n"
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

class InventoryVisualization:
    def __init__(self, dates, inventory_level_ss, inventory_level_sq, inventory_level_labelling, inventory_level_out,
                 po_ss, po_sq, po_out, po_labelling, po_reaching_ss, po_reaching_sq, po_reaching_out, po_reaching_labelling):
        self.dates = dates
        self.inventory_level_ss = inventory_level_ss
        self.inventory_level_sq = inventory_level_sq
        self.inventory_level_labelling = inventory_level_labelling
        self.inventory_level_out = inventory_level_out
        self.po_ss = po_ss
        self.po_sq = po_sq
        self.po_out = po_out
        self.po_labelling = po_labelling
        self.po_reaching_ss = po_reaching_ss
        self.po_reaching_sq = po_reaching_sq
        self.po_reaching_out = po_reaching_out
        self.po_reaching_labelling = po_reaching_labelling

    def plot_simulation(self, title="Inventory Simulation"):
        plt.figure(figsize=(15, 8))

        plt.plot(self.dates, self.inventory_level_ss, label="Safety Stock (SS) Policy", linestyle='-', color='blue')
        plt.plot(self.dates, self.inventory_level_sq, label="Fixed Order-Quantity Policy", linestyle='-', color='orange')
        plt.plot(self.dates, self.inventory_level_labelling, label="Labelling Policy", linestyle='-', color='green')
        plt.plot(self.dates, self.inventory_level_out, label="Order-Up-To Policy", linestyle='-', color='purple')

        for i, date in enumerate(self.dates):
            if i in self.po_out.keys():
                plt.axvline(date, color='orange', linestyle='--', alpha=0.3)
            if i in self.po_labelling.keys():
                plt.axvline(date, color='green', linestyle='--', alpha=0.3)
            if i in self.po_reaching_out:
                plt.axvline(date, color='purple', linestyle='--', alpha=0.3)
            if i in self.po_reaching_labelling:
                plt.axvline(date, color='purple', linestyle='--', alpha=0.3)

        plt.xlabel("Date")
        plt.ylabel("Inventory Level")
        plt.title(title)
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def export_data(self, filename='inventory_data.csv'):
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([
                'Date',
                'SS Policy Inventory Level',
                'Fixed Order-Quantity Policy Inventory Level',
                'Labelling Policy Inventory Level',
                'Order-Up-To Policy Inventory Level',
                'SS Policy Reorder Quantity',
                'Fixed Order-Quantity Reorder Quantity',
                'Order-Up-To Policy Reorder Quantity',
                'Labelling Reorder Quantity',
                'SS Policy Reorder Date',
                'Fixed Order-Quantity Reorder Date',
                'Order-Up-To Policy Reorder Date',
                'Labelling Reorder Date',
                'SS Policy Delivery Date',
                'Fixed Order-Quantity Policy Delivery Date',
                'Order-Up-To Policy Delivery Date',
                'Labelling Delivery Date'
            ])

            reorder_dates_ss = set(self.po_ss.keys())
            reorder_dates_sq = set(self.po_sq.keys())
            reorder_dates_out = set(self.po_out.keys())
            reorder_dates_labelling = set(self.po_labelling.keys())
            delivery_dates_ss = set(self.po_reaching_ss)
            delivery_dates_sq = set(self.po_reaching_sq)
            delivery_dates_out = set(self.po_reaching_out)
            delivery_dates_labelling = set(self.po_labelling.keys())

            for i, date in enumerate(self.dates):
                reorder_qty_ss = self.po_ss.get(i, 0)
                reorder_qty_sq = self.po_sq.get(i, 0)
                reorder_qty_out = self.po_out.get(i, 0)
                reorder_qty_labelling = self.po_labelling.get(i, 0)

                csvwriter.writerow([
                    date.strftime('%Y-%m-%d'),
                    self.inventory_level_ss[i],
                    self.inventory_level_sq[i],
                    self.inventory_level_labelling[i],
                    self.inventory_level_out[i],
                    reorder_qty_ss if i in reorder_dates_ss else '',
                    reorder_qty_sq if i in reorder_dates_sq else '',
                    reorder_qty_out if i in reorder_dates_out else '',
                    reorder_qty_labelling if i in reorder_dates_labelling else '',
                    'Yes' if i in reorder_dates_ss else 'No',
                    'Yes' if i in reorder_dates_sq else 'No',
                    'Yes' if i in reorder_dates_out else 'No',
                    'Yes' if i in reorder_dates_labelling else 'No',
                    'Yes' if i in delivery_dates_ss else 'No',
                    'Yes' if i in delivery_dates_sq else 'No',
                    'Yes' if i in delivery_dates_out else 'No',
                    'Yes' if i in delivery_dates_labelling else 'No',
                ])

        print(f"Inventory data exported to {filename}")

def main():
    distributions = {
        'normal': {'mu': 10, 'std': 6},
        'gamma': {'shape': 2, 'scale': 5},
        'lognormal': {'mu': 2, 'sigma': 0.5},
        'poisson': {'lambda': 10}
    }

    for demand_dist, demand_params in distributions.items():
        print(f"\nSimulation with {demand_dist.capitalize()} Demand Distribution")

        simulation = InventorySimulation(
            demand_distribution=demand_dist,
            demand_params=demand_params,
            lead_time_distribution='normal',
            lead_time_params={'mu': 5, 'std': 1},
            b=9,
            h=1,
            fixed_reorder_quantity=75,
            order_up_to_point=200
        )

        results = simulation.run_simulation()

        visualization = InventoryVisualization(
            results['dates'],
            results['inventory_level_ss'],
            results['inventory_level_sq'],
            results['inventory_level_labelling'],
            results['inventory_level_out'],
            results['po_ss'],
            results['po_sq'],
            results['po_out'],
            results['po_labelling'],
            results['po_reaching_ss'],
            results['po_reaching_sq'],
            results['po_reaching_out'],
            results['po_reaching_labelling']
        )

        visualization.plot_simulation(title=f"Inventory Replenishment - {demand_dist.capitalize()} Demand")
        visualization.export_data(filename=f"inventory_data_{demand_dist}.csv")

        analysis = InventoryAnalysis(
            results['inventory_level_ss'],
            results['inventory_level_sq'],
            results['inventory_level_labelling'],
            results['inventory_level_out'],
            results['future_demand']
        )

        analysis_results = analysis.analyze()
        report = analysis.format_report(analysis_results)
        print(report)

if __name__ == "__main__":
    main()

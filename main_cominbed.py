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

# Generate demand using specified probability distribution
def generate_demand(distribution, params, size):
    distributions = {
        'normal': stats.norm.rvs,
        'gamma': stats.gamma.rvs,
        # 'weibull': stats.weibull_min.rvs,
        'poisson': stats.poisson.rvs
    }

    # Validate distribution
    if distribution not in distributions:
        raise ValueError(f"Unsupported distribution: {distribution}")

    # Distribution-specific parameter mapping
    if distribution == 'normal':
        return np.abs(distributions[distribution](loc=params.get('mu', 10),
                                                  scale=params.get('std', 6),
                                                  size=size))
    elif distribution == 'gamma':
        return distributions[distribution](a=params.get('shape', 2),
                                           scale=params.get('scale', 1),
                                           size=size)
    # elif distribution == 'weibull':
    #     return distributions[distribution](c=params.get('shape', 1),
    #                                        scale=params.get('scale', 1),
    #                                        size=size)
    elif distribution == 'poisson':
        return distributions[distribution](mu=params.get('lambda', 2),
                                           size=size)


# Generate lead time using specified probability distribution
def generate_lead_time(distribution, params, size):
    # Use the same distributions as demand generation
    return np.abs(generate_demand(distribution, params, size)).astype(int)


def run_inventory_simulation(
        initial_inventory=100,
        horizon=60,
        reorder_days=[1, 4],  # Reorder days (i.e. Tuesday = 1, Friday = 4)
        service_level=0.99,
        demand_distribution='normal',
        demand_params=None,
        lead_time_distribution='normal',
        lead_time_params=None,
        b=9,
        h=1,
        fixed_reorder_quantity=100,
        order_up_to_point=200
):
    # Set default parameters if not provided
    if demand_params is None:
        demand_params = {'mu': 10, 'std': 6}
    if lead_time_params is None:
        lead_time_params = {'mu': 5, 'std': 1}

    set_random_seed()

    # Create date array
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(horizon)]

    # Generate reorder points for specified days
    reorder_point_arr = np.zeros(horizon, dtype=bool)
    for i, date in enumerate(dates):
        reorder_point_arr[i] = date.weekday() in reorder_days

    n_reorder_pts = len(np.where(reorder_point_arr)[0])  # Number of reorder points in the review period

    # Generate demand and lead time histories
    history_lead_time = generate_lead_time(lead_time_distribution, lead_time_params, 50)
    history_demand = generate_demand(demand_distribution, demand_params, 50)
    future_lead_time = generate_lead_time(lead_time_distribution, lead_time_params, n_reorder_pts)
    future_demand = generate_demand(demand_distribution, demand_params, horizon)

    # Calculate safety stock (assuming historical data is used for calculation)
    safety_stock = get_safety_stock(history_lead_time, history_demand, service_level)

    # SS Policy Inventory Replenishment
    inventory_level_ss = np.zeros(horizon)
    inventory_level_ss[0] = initial_inventory
    reorder_ptr = 0  # the pointer to count number of times reorder happens within review period
    po_ss = dict()
    po_reaching_ss = []
    pending_order_ss = None  # Track pending order for SS policy

    # Fixed Order,Quantity Policy Inventory Replenishment
    inventory_level_sq = np.zeros(horizon)
    inventory_level_sq[0] = initial_inventory
    po_sq = dict()
    po_reaching_sq = []
    pending_order_sq = None

    # Order-Up-To Policy Inventory Replenishment
    inventory_level_out = np.zeros(horizon)
    inventory_level_out[0] = initial_inventory
    po_out = dict()
    po_reaching_out = []
    pending_order_out = None

    # Labelling Policy Inventory Replenishment
    inventory_level_labelling = np.zeros(horizon)
    inventory_level_labelling[0] = initial_inventory

    # Generate po reaching for labelling policy
    po_reaching_labelling = list(np.where(reorder_point_arr)[0] + np.int32(np.ceil(future_lead_time)))
    po_reaching_labelling.append(po_reaching_labelling[-1] + np.ceil(np.diff(po_reaching_labelling).mean()))

    # Add tracking for pending orders in labelling policy
    pending_order_labelling = None
    po_labelling = dict()  # Track reorder quantities
    actual_po_reaching_labelling = []  # Track actual delivery dates

    # Initialize variables for different policies
    inventory_level_ss = np.zeros(horizon)
    inventory_level_sq = np.zeros(horizon)
    inventory_level_labelling = np.zeros(horizon)
    inventory_level_out = np.zeros(horizon)

    inventory_level_ss[0] = initial_inventory
    inventory_level_sq[0] = initial_inventory
    inventory_level_labelling[0] = initial_inventory
    inventory_level_out[0] = initial_inventory

    # Reorder pointers for different policies
    reorder_ptr = 0
    reorder_ptr_sq = 0
    reorder_ptr_out = 0

    # Tracking variables for different policies
    po_ss = dict()
    po_sq = dict()
    po_out = dict()
    po_reaching_ss = []
    po_reaching_sq = []
    po_reaching_out = []
    pending_order_ss = None
    pending_order_sq = None
    pending_order_out = None

    for i in range(1, horizon):
        reorder_qty = 0
        # SS Policy Implementation
        inventory_level_ss[i] = inventory_level_ss[i - 1] + inventory_level_ss[i] - future_demand[i]

        if pending_order_ss and i >= pending_order_ss['delivery_date']:
            pending_order_ss = None

        if reorder_point_arr[i] and pending_order_ss is None:
            interval = 7
            demand_forecast = abs(future_demand[reorder_ptr] + np.random.normal(0, 0))

            # Amended from mu_lead_time to future_lead_time[reorder_ptr] for this version
            delta1 = inventory_level_ss[i] - demand_forecast * future_lead_time[reorder_ptr] - safety_stock
            if delta1 < 0:
                reorder_qty = -delta1

            delta2 = inventory_level_ss[i] - demand_forecast * interval - safety_stock
            if delta2 < 0:
                reorder_qty = max(reorder_qty, -delta2)

            if reorder_qty > 0:
                delivery_date = i + int(np.ceil(future_lead_time[reorder_ptr]))
                pending_order_ss = {
                    'quantity': reorder_qty,
                    'delivery_date': delivery_date
                }

                if delivery_date < horizon:
                    inventory_level_ss[delivery_date] += reorder_qty
                    po_reaching_ss.append(delivery_date)
                    po_ss[i] = reorder_qty

            reorder_ptr += 1

        # Fixed Order-Quantity Policy Implementation
        inventory_level_sq[i] = inventory_level_sq[i - 1] + inventory_level_sq[i] - future_demand[i]

        if pending_order_sq and i >= pending_order_sq['delivery_date']:
            pending_order_sq = None

        if reorder_point_arr[i] and pending_order_sq is None:
            reorder_qty = fixed_reorder_quantity
            delivery_date = i + int(np.ceil(future_lead_time[reorder_ptr_sq]))

            pending_order_sq = {
                'quantity': reorder_qty,
                'delivery_date': delivery_date
            }

            if delivery_date < horizon:
                inventory_level_sq[delivery_date] += reorder_qty
                po_reaching_sq.append(delivery_date)
                po_sq[i] = reorder_qty

            reorder_ptr_sq += 1

        # Order-Up-To Policy Implementation
        inventory_level_out[i] = inventory_level_out[i - 1] + inventory_level_out[i] - future_demand[i]

        if pending_order_out and i >= pending_order_out['delivery_date']:
            pending_order_out = None

        if reorder_point_arr[i] and pending_order_out is None:
            inventory_before_demand = inventory_level_out[i - 1]
            reorder_qty = max(0, order_up_to_point - inventory_before_demand)

            if reorder_qty > 0:
                delivery_date = i + int(np.ceil(future_lead_time[reorder_ptr_out]))
                pending_order_out = {
                    'quantity': reorder_qty,
                    'delivery_date': delivery_date
                }

                if delivery_date < horizon:
                    inventory_level_out[delivery_date] += reorder_qty
                    po_reaching_out.append(delivery_date)
                    po_out[i] = reorder_qty

            reorder_ptr_out += 1

    # Labelling Policy with pending order constraint
    t = 0
    for i in range(1, horizon):
        # Deduct inventory based on demand
        inventory_level_labelling[i] = inventory_level_labelling[i - 1] + inventory_level_labelling[i] - future_demand[
            i]

        # Check if pending order is delivered
        if pending_order_labelling and i >= pending_order_labelling['delivery_date']:
            pending_order_labelling = None

        if reorder_point_arr[i] and t < len(po_reaching_labelling):
            # Skip if there's a pending order
            if pending_order_labelling is not None:
                continue

            vm = po_reaching_labelling[t]

            # Ensure vm is within the horizon
            if vm >= horizon:
                break

            delta_t = int(
                np.floor(b * (po_reaching_labelling[t + 1] - po_reaching_labelling[t]) / (h + b)))

            # Ensure we don't exceed the horizon when calculating cumulative demand
            delta_t = min(delta_t, horizon - vm)

            # Calculate cumulative demand ensuring we don't exceed array bounds
            cum_demand = np.sum(future_demand[vm:min(vm + delta_t, horizon)])

            # Calculate the optimal reorder quantity
            optimal_qty = max(cum_demand + np.sum(future_demand[i:vm]) - inventory_level_labelling[i], 0)

            if optimal_qty > 0:
                # Set pending order
                pending_order_labelling = {
                    'quantity': optimal_qty,
                    'delivery_date': vm
                }

                # Update inventory level at delivery date
                if vm < horizon:
                    inventory_level_labelling[vm] += optimal_qty
                    actual_po_reaching_labelling.append(vm)
                    po_labelling[i] = optimal_qty

            t += 1

    return (
        inventory_level_ss,
        inventory_level_sq,
        inventory_level_labelling,
        inventory_level_out,
        po_ss,
        po_sq,
        po_out,
        po_labelling,
        po_reaching_ss,
        po_reaching_sq,
        po_reaching_out,
        po_reaching_labelling,
        dates,
        future_demand
    )


def plot_inventory_simulation(
    inventory_level_ss,
    inventory_level_sq,
    inventory_level_labelling,
    inventory_level_out,  # Order-Up-To Policy inventory levels
    dates,
    po_ss,
    po_sq,
    po_out,
    po_labelling,
    po_reaching_ss,
    po_reaching_sq,
    po_reaching_out,
    po_reaching_labelling,
    title="Inventory Simulation"
):

    plt.figure(figsize=(15, 8))

    # Plot inventory levels for each policy
    plt.plot(dates, inventory_level_ss, label="Safety Stock (SS) Policy", linestyle='-', color='blue')
    plt.plot(dates, inventory_level_sq, label="Fixed Order-Quantity Policy", linestyle='-', color='orange')
    plt.plot(dates, inventory_level_labelling, label="Labelling Policy", linestyle='-', color='green')
    plt.plot(dates, inventory_level_out, label="Order-Up-To Policy", linestyle='-', color='purple')

    # Highlight reorder points and delivery dates
    for i, date in enumerate(dates):
        # Reorder points
        # if i in po_ss.keys():
            # plt.axvline(date, color='orange', linestyle='--', alpha=0.3,
            #             label='Reorder Point' if i == list(po_ss.keys())[0] else "")
        # if i in po_sq.keys():
        #     plt.axvline(date, color='orange', linestyle='--', alpha=0.3)
        if i in po_out.keys():
            plt.axvline(date, color='orange', linestyle='--', alpha=0.3)
        if i in po_labelling.keys():
            plt.axvline(date, color='green', linestyle='--', alpha=0.3)

        # Delivery dates
        # if i in po_reaching_ss:
        #     plt.axvline(date, color='purple', linestyle='--', alpha=0.3,
        #                 label='Delivery Date' if i == po_reaching_ss[0] else "")
        # if i in po_reaching_sq:
        #     plt.axvline(date, color='purple', linestyle='--', alpha=0.3)
        if i in po_reaching_out:
            plt.axvline(date, color='purple', linestyle='--', alpha=0.3)
        if i in po_reaching_labelling:
            plt.axvline(date, color='purple', linestyle='--', alpha=0.3)
        # if i > 0 and inventory_level_labelling[i] > inventory_level_labelling[i - 1]:
        #     plt.axvline(date, color='purple', linestyle='--', alpha=0.3)

    # Add labels, legend, and title
    plt.xlabel("Date")
    plt.ylabel("Inventory Level")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()

    # Show plot
    plt.show()




def export_inventory_data(
    dates,
    inventory_level_ss,
    inventory_level_sq,
    inventory_level_labelling,
    inventory_level_out,
    po_ss,
    po_sq,
    po_out,
    po_labelling,
    po_reaching_ss,
    po_reaching_sq,
    po_reaching_out,
    po_reaching_labelling,
    filename='inventory_data.csv'
):
    """
    Export inventory data to a CSV file
    """
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow([
            'Date',
            'SS Policy Inventory Level',
            'Fixed Order,Quantity Policy Inventory Level',
            'Labelling Policy Inventory Level',
            'Order-Up-To Policy Inventory Level',
            'SS Policy Reorder Quantity',
            'Fixed Order,Quantity Reorder Quantity',
            'Order-Up-To Policy Reorder Quantity',
            'Labelling Reorder Quantity',
            'SS Policy Reorder Date',
            'Fixed Order,Quantity Reorder Date',
            'Order-Up-To Policy Reorder Date',
            'Labelling Reorder Date',
            'SS Policy Delivery Date',
            'Fixed Order,Quantity Policy Delivery Date',
            'Order-Up-To Policy Delivery Date',
            'Labelling Delivery Date'
        ])

        # Create sets of reorder and delivery dates
        reorder_dates_ss = set(po_ss.keys())
        reorder_dates_sq = set(po_sq.keys())
        reorder_dates_out = set(po_out.keys())
        reorder_dates_labelling = set(po_labelling.keys())
        delivery_dates_ss = set(po_reaching_ss)
        delivery_dates_sq = set(po_reaching_sq)
        delivery_dates_out = set(po_reaching_out)
        delivery_dates_labelling = set(po_labelling.keys())

        # Write data rows
        for i, date in enumerate(dates):
            # Get reorder quantities and check dates
            reorder_qty_ss = po_ss.get(i, 0)
            reorder_qty_sq = po_sq.get(i, 0)
            reorder_qty_out = po_out.get(i, 0)
            reorder_qty_labelling = po_labelling.get(i, 0)

            csvwriter.writerow([
                date.strftime('%Y-%m-%d'),
                inventory_level_ss[i],
                inventory_level_sq[i],
                inventory_level_labelling[i],
                inventory_level_out[i],
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


def main():
    distributions = {
        'normal': {'mu': 10, 'std': 6},
        'gamma': {'shape': 2, 'scale': 5},
        # 'weibull': {'shape': 2, 'scale': 5},
        'poisson': {'lambda': 10}
    }

    for demand_dist, demand_params in distributions.items():
        print(f"\nSimulation with {demand_dist.capitalize()} Demand Distribution")

        # Run simulation
        (inventory_level_ss,
         inventory_level_sq,
         inventory_level_labelling,
         inventory_level_out,
         po_ss,
         po_sq,
         po_out,
         po_labelling,
         po_reaching_ss,
         po_reaching_sq,
         po_reaching_out,
         po_reaching_labelling,
         dates,
         future_demand) = run_inventory_simulation(
            demand_distribution=demand_dist,
            demand_params=demand_params,
            lead_time_distribution='normal',
            lead_time_params={'mu': 5, 'std': 1},
            b=9,
            h=1,
            fixed_reorder_quantity=75,  # Fixed reorder quantity for Fixed Order,Quantity policy
            order_up_to_point=200  # Target inventory level for Order-Up-To Policy
        )

        # Plot results
        plot_inventory_simulation(
            inventory_level_ss,
            inventory_level_sq,
            inventory_level_labelling,
            inventory_level_out,
            dates,
            po_ss,
            po_sq,
            po_out,
            po_labelling,
            po_reaching_ss,
            po_reaching_sq,
            po_reaching_out,
            po_reaching_labelling,
            title=f"Inventory Replenishment - {demand_dist.capitalize()} Demand"
        )

        # Export data
        export_inventory_data(
            dates,
            inventory_level_ss,
            inventory_level_sq,
            inventory_level_labelling,
            inventory_level_out,
            po_ss,
            po_sq,
            po_out,
            po_labelling,
            po_reaching_ss,
            po_reaching_sq,
            po_reaching_out,
            po_reaching_labelling,
            filename=f"inventory_data_{demand_dist}.csv"
        )

        # Perform analysis
        results = analyze_inventory_policies(
            inventory_level_ss,
            inventory_level_sq,
            inventory_level_labelling,
            inventory_level_out,
            future_demand
        )

        # Generate and print the report
        report = format_analysis_report(results)
        print(report)

if __name__ == "__main__":
    main()
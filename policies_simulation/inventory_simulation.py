import numpy as np
from datetime import datetime, timedelta
import scipy.stats as stats
from inventory_policies import PTOPolicy, FixedOrderQuantityPolicy, OrderUpToPolicy, E2EPolicy  # Import the policy classes

class InventorySimulation:
    def __init__(self, initial_inventory=100,
                 horizon=60,
                 reorder_days=[4],
                 service_level=0.99,
                 demand_distribution='normal',
                 demand_params=None,
                 lead_time_distribution='normal',
                 lead_time_params=None,
                 b=9,
                 h=1,
                 fixed_reorder_quantity=100,
                 order_up_to_point=200):
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

        pto_policy = PTOPolicy(self.initial_inventory, safety_stock, self.service_level)
        fixed_policy = FixedOrderQuantityPolicy(self.initial_inventory, self.fixed_reorder_quantity)
        upto_policy = OrderUpToPolicy(self.initial_inventory, self.order_up_to_point)
        po_reaching_e2e = list(np.where(reorder_point_arr)[0] + np.ceil(future_lead_time).astype(int))
        po_reaching_e2e.append(po_reaching_e2e[-1] + int(np.diff(po_reaching_e2e).mean()))
        e2e_policy = E2EPolicy(self.initial_inventory, self.b, self.h)

        policies = {
            'pto': pto_policy,
            'fixed': fixed_policy,
            'upto': upto_policy,
            'e2e': e2e_policy
        }

        for i in range(1, self.horizon):
            for name, policy in policies.items():
                policy.receive_shipment(i)
                policy.update_inventory(i, future_demand[i])
                if name == 'e2e':
                    reorder_qty = policy.reorder(i, reorder_point_arr[i], future_demand, future_lead_time, self.horizon, po_reaching_e2e)
                else:
                    reorder_qty = policy.reorder(i, reorder_point_arr[i], future_demand, future_lead_time, self.horizon)

        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(self.horizon)]
        results = {
            'dates': dates,
            'future_demand': future_demand,
            'inventory_level_pto': pto_policy.inventory_level,
            'inventory_level_sq': fixed_policy.inventory_level,
            'inventory_level_out': upto_policy.inventory_level,
            'inventory_level_e2e': e2e_policy.inventory_level,
            'po_pto': pto_policy.po,
            'po_sq': fixed_policy.po,
            'po_out': upto_policy.po,
            'po_e2e': e2e_policy.po,
            'po_reaching_pto': pto_policy.po_reaching,
            'po_reaching_sq': fixed_policy.po_reaching,
            'po_reaching_out': upto_policy.po_reaching,
            'po_reaching_e2e': e2e_policy.actual_po_reaching,
            'po_reaching_e2e_schedule': po_reaching_e2e
        }

        return results

    def get_safety_stock(self, history_lead_time, history_demand, service_level):
        return np.mean(history_demand) * np.mean(history_lead_time) * service_level
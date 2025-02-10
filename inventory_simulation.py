import numpy as np
from datetime import datetime, timedelta
from data_generation import generate_demand, generate_lead_time, set_random_seed
from invutils import get_safety_stock

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
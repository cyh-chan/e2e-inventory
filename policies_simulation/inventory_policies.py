import numpy as np

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

class PTOPolicy(InventoryPolicy):
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

class E2EPolicy(InventoryPolicy):
    def __init__(self, initial_inventory, b, h):
        super().__init__(initial_inventory)
        self.b = b
        self.h = h
        self.t = 0
        self.actual_po_reaching = []

    def reorder(self, day, reorder_point, future_demand, future_lead_time, horizon, po_reaching_e2e):
        if self.pending_order:
            return 0

        if not reorder_point:
            return 0

        if self.t >= len(po_reaching_e2e):
            return 0

        vm = po_reaching_e2e[self.t]
        if vm >= horizon:
            return 0

        delta_t = int(np.floor(self.b * (po_reaching_e2e[self.t + 1] - po_reaching_e2e[self.t]) / (self.h + self.b)))
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
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from invutils.invutils import get_demand, get_lead_time, get_safety_stock, get_reorder_point, get_target_inventory
from inventory_simulation import InventorySimulation
from inventory_visualization import InventoryVisualization
from inventory_analysis import InventoryAnalysis

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
            results['inventory_level_pto'],
            results['inventory_level_sq'],
            results['inventory_level_e2e'],
            results['inventory_level_out'],
            results['po_pto'],
            results['po_sq'],
            results['po_out'],
            results['po_e2e'],
            results['po_reaching_pto'],
            results['po_reaching_sq'],
            results['po_reaching_out'],
            results['po_reaching_e2e']
        )

        visualization.plot_simulation(title=f"Inventory Replenishment - {demand_dist.capitalize()} Demand")
        visualization.export_data(filename=f"inventory_data_{demand_dist}.csv")

        analysis = InventoryAnalysis(
            results['inventory_level_pto'],
            results['inventory_level_sq'],
            results['inventory_level_e2e'],
            results['inventory_level_out'],
            results['future_demand']
        )

        analysis_results = analysis.analyze()
        report = analysis.format_report(analysis_results)
        print(report)

if __name__ == "__main__":
    main()
# File structure:
# main.py
# data_generation.py
# inventory_simulation.py
# analysis.py
# visualization.py
# invutils.py

import numpy as np
from inventory_simulation import run_inventory_simulation
from visualization import plot_inventory_simulation, export_inventory_data
from analysis import analyze_inventory_policies, format_analysis_report

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
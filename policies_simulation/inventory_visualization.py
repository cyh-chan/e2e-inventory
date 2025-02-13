# inventory_visualization.py
import matplotlib.pyplot as plt
import csv
from datetime import datetime

class InventoryVisualization:
    def __init__(self, dates, inventory_level_pto, inventory_level_sq, inventory_level_e2e, inventory_level_out,
                 po_pto, po_sq, po_out, po_e2e, po_reaching_pto, po_reaching_sq, po_reaching_out, po_reaching_e2e):
        self.dates = dates
        self.inventory_level_pto = inventory_level_pto
        self.inventory_level_sq = inventory_level_sq
        self.inventory_level_e2e = inventory_level_e2e
        self.inventory_level_out = inventory_level_out
        self.po_pto = po_pto
        self.po_sq = po_sq
        self.po_out = po_out
        self.po_e2e = po_e2e
        self.po_reaching_pto = po_reaching_pto
        self.po_reaching_sq = po_reaching_sq
        self.po_reaching_out = po_reaching_out
        self.po_reaching_e2e = po_reaching_e2e

    def plot_simulation(self, title="Inventory Simulation"):
        plt.figure(figsize=(15, 8))

        plt.plot(self.dates, self.inventory_level_pto, label="Predict-Then-Optimize (PTO) Policy", linestyle='-', color='blue')
        plt.plot(self.dates, self.inventory_level_sq, label="Fixed Order-Quantity Policy", linestyle='-', color='orange')
        plt.plot(self.dates, self.inventory_level_e2e, label="E2E Policy", linestyle='-', color='green')
        plt.plot(self.dates, self.inventory_level_out, label="Order-Up-To Policy", linestyle='-', color='purple')

        for i, date in enumerate(self.dates):
            if i in self.po_out.keys():
                plt.axvline(date, color='orange', linestyle='--', alpha=0.3)
            if i in self.po_e2e.keys():
                plt.axvline(date, color='green', linestyle='--', alpha=0.3)
            if i in self.po_reaching_out:
                plt.axvline(date, color='purple', linestyle='--', alpha=0.3)
            if i in self.po_reaching_e2e:
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
                'PTO Policy Inventory Level',
                'Fixed Order-Quantity Policy Inventory Level',
                'E2E Policy Inventory Level',
                'Order-Up-To Policy Inventory Level',
                'PTO Policy Reorder Quantity',
                'Fixed Order-Quantity Reorder Quantity',
                'Order-Up-To Policy Reorder Quantity',
                'E2E Reorder Quantity',
                'PTO Policy Reorder Date',
                'Fixed Order-Quantity Reorder Date',
                'Order-Up-To Policy Reorder Date',
                'E2E Reorder Date',
                'PTO Policy Delivery Date',
                'Fixed Order-Quantity Policy Delivery Date',
                'Order-Up-To Policy Delivery Date',
                'E2E Delivery Date'
            ])

            reorder_dates_pto = set(self.po_pto.keys())
            reorder_dates_sq = set(self.po_sq.keys())
            reorder_dates_out = set(self.po_out.keys())
            reorder_dates_e2e = set(self.po_e2e.keys())
            delivery_dates_pto = set(self.po_reaching_pto)
            delivery_dates_sq = set(self.po_reaching_sq)
            delivery_dates_out = set(self.po_reaching_out)
            delivery_dates_e2e = set(self.po_e2e.keys())

            for i, date in enumerate(self.dates):
                reorder_qty_pto = self.po_pto.get(i, 0)
                reorder_qty_sq = self.po_sq.get(i, 0)
                reorder_qty_out = self.po_out.get(i, 0)
                reorder_qty_e2e = self.po_e2e.get(i, 0)

                csvwriter.writerow([
                    date.strftime('%Y-%m-%d'),
                    self.inventory_level_pto[i],
                    self.inventory_level_sq[i],
                    self.inventory_level_e2e[i],
                    self.inventory_level_out[i],
                    reorder_qty_pto if i in reorder_dates_pto else '',
                    reorder_qty_sq if i in reorder_dates_sq else '',
                    reorder_qty_out if i in reorder_dates_out else '',
                    reorder_qty_e2e if i in reorder_dates_e2e else '',
                    'Yes' if i in reorder_dates_pto else 'No',
                    'Yes' if i in reorder_dates_sq else 'No',
                    'Yes' if i in reorder_dates_out else 'No',
                    'Yes' if i in reorder_dates_e2e else 'No',
                    'Yes' if i in delivery_dates_pto else 'No',
                    'Yes' if i in delivery_dates_sq else 'No',
                    'Yes' if i in delivery_dates_out else 'No',
                    'Yes' if i in delivery_dates_e2e else 'No',
                ])

        print(f"Inventory data exported to {filename}")
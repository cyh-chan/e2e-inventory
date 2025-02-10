import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import csv

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
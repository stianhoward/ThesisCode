"""
Plotting tools that are frequently used

"""

import matplotlib.patches as mpatches
import matplotlib.lines as mlines

def insert_plot(ax, data, param_dict, addterms=True):
    """
    insert_rawplot(ax, data, param_dict)
    Insert the rawplot to the provided axis
    ax: matplotlib axis object
    data: a datafram containing 'Oil Fraction", "Gas Fraction", "Water Fraction", "Rate", and
        "DPtotal".
    param_dict: a dictionary of paramters to pass to the plot functions
    """

    # Group data by Oil Fraction, Gas Fraction and Water Fraction
    groups = data.groupby(["Oil Fraction", "Gas Fraction","Water Fraction"])

    if not 'color' in param_dict:
        recolor = True
    else:
        recolor = False

    # Iterate through the fluid fraction and plot idividual curves
    for name, group in groups:
        if recolor:
            param_dict['color'] = name
        group = group.sort_values('Rate')
        ax.plot(group['Rate'], group['DPtotal'], **param_dict)

    # Create a reasonable legend
    oil_patch = mpatches.Patch(color=(1.0,0,0), label='100% oil')
    gas_patch = mpatches.Patch(color=(0,1.0,0), label='100% gas')
    water_patch = mpatches.Patch(color=(0,0,1.0), label='100% water')

    if addterms:
        solid = mlines.Line2D([], [], color='black', marker='s', linestyle='solid', markersize=0, label='Test Data')
        dashed = mlines.Line2D([], [], color='black', marker='s', linestyle='--', markersize=0, label='Prediction')
        ax.legend(handles=[oil_patch, gas_patch, water_patch, solid, dashed])
    else:
        ax.legend(handles=[oil_patch, gas_patch, water_patch])

    # Figure titles
    ax.set_xlabel("Fluid Flow rate (Rm3/day)")
    ax.set_ylabel("Pressure Drop (bar)")


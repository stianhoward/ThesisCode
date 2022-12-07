"""
Plotting for a model on top of raw data
"""

import matplotlib.pyplot as plt

import plot_tools
from torch import tensor


def main():
    pass


def create_modelfigure(func, func_params, raw_data, toTensor=False, subPlots = 111, fig = None, title = "Flow rate vs Pressure Drop"):
    """
    create_modelplot(func, func_params, raw_data)
    Generate a figure for displaying results of a model
    func: function call containing the model of format func([independent variables], [params])
    func_params: array of parameters to be passed to func as the fitted model
    raw_data: the raw training data on which the model is relevant
    """

    if not fig:
        fig = plt.figure()
    ax = fig.add_subplot(subPlots)
    ax.set_title(title)

    # add raw data plots
    param_dict = {}
    plot_tools.insert_plot(ax, raw_data, param_dict)

    # manage and calculate data values for model
    dc = raw_data.copy()
    #print(tensor(dc.values.tolist()))
    if toTensor:
        dc['DPtotal'] = dc.apply(lambda x: func(tensor(retrieve_independent(x))).detach().numpy(), axis=1)
    else:
        dc['DPtotal'] = dc.apply(lambda x: func(retrieve_independent(x), *func_params),axis=1)

    # Add the new points as well
    param_dict = {'linestyle':'dashed'}
    plot_tools.insert_plot(ax, dc, param_dict)

    return fig


def retrieve_independent(df):
    """
    retrieve_indedendent(df)
    Extracts the independent variables of interest and return them in a python list
    df: Pandas DataFrame with all required values
    Return: python list containing the independent variables
    """
    #print(df[['Rate', 'Oil Fraction', 'rhoo', 'muo', 'Gas Fraction', 'rhog', 'mug', 'Water Fraction', 'rhow', 'muw']].values.tolist())
    ind_vars = df[['Rate', 'Oil Fraction', 'rhoo', 'muo', 'Gas Fraction', 'rhog', 'mug', 'Water Fraction', 'rhow', 'muw']].values.tolist()
    return ind_vars

if __name__ == "__main__":
    main()

"""
raw_plots.py
Generate visualization of raw data for pressure drop characterization data
"""

import plot_tools

import os

import matplotlib.pyplot as plt
import pandas as pd


# Path to raw data
RAW_DATA_PATH = '../Thesis_Data/completeData.csv'
EXPORT_DIR = './plots/'

def main(raw_path = RAW_DATA_PATH, exp_dir = EXPORT_DIR):
    # load the raw imput data
    raw_data = load_data(raw_path)

    # create export directory if doesn't exist
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    fig = create_rawfigure(raw_data, 145)
    plt.savefig(os.path.join(exp_dir, "raw_145.jpg"))
    fig = create_rawfigure(raw_data, 150)
    plt.savefig(os.path.join(exp_dir, "raw_150.jpg"))


def load_data(csv_path):
    """
    load_data(csv_path)
    load the data from csv file at csv_path into a pandas data structure
    """

    data = pd.read_csv(csv_path, skiprows=[1])
    #print(data.loc[data['B.H.Pressure'] == 145])
    return data


def create_rawfigure(data, bhpressure):
    """
    create_plots(data, export_path)
    Generate the raw data plots and export them to export_path
    data: pandas dataframe with 'Rate', 'DPtotal', 'Water Fraction', 'Oil Fraction', and 'Gas Fraction'
    export_path: string to the export path. Potentially this should be changed to always be './plots'
    """

    # grab only the first selection of data with requested B.H pressure
    dataBH = data.loc[data['B.H.Pressure'] == bhpressure]
    #dataBH = data

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"Pressure Drop vs Fluid Flow Rate at {bhpressure}bar Fluid Pressure")

    plot_tools.insert_plot(ax, dataBH, {}, addterms=False)

    return fig


if __name__ == "__main__":
    main()

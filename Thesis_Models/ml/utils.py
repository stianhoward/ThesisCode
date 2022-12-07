"""
Utility functions for machine learning tools

"""
import pandas as pd
import torch
from torch.utils.data import DataLoader

"""
getDataLoader

"""
def getDataLoader(path, shuffle=True):
    data = load_pd_data(path)

    x_data = data[['Rate', 'Oil Fraction', 'rhoo', 'muo', 'Gas Fraction', 'rhog', 'mug', 'Water Fraction', 'rhow', 'muw']]
    y_data = data['DPtotal']

    out_data = []
    for x,y in zip(x_data.values.tolist(), y_data.values.tolist()):
        out_data.append([torch.tensor(x),torch.tensor([y])])

    dl = DataLoader(out_data, shuffle=shuffle, batch_size=5, num_workers=8)
    return dl


"""
getOneValue(path, line=0)
Retrieve one value from the data set and return it as a python list
path: pata to the data CSV
line: which line in the dataset should be extracted

return: Python list of the input data values
"""
def getOneValue(path, line=0):
    data = load_pd_data(path)

    x_data = data[['Rate', 'Oil Fraction', 'rhoo', 'muo', 'Gas Fraction', 'rhog', 'mug', 'Water Fraction', 'rhow', 'muw']]
    ret_data = x_data.values.tolist()[line]
    ret_data = torch.Tensor(ret_data)
    return ret_data


"""
load_pd_data(csv_path)

Load the data to be fitted into the file, and return it as a pandas array
csv_path: path to the csv file to be loaded
"""
def load_pd_data(csv_path):
    data = pd.read_csv(csv_path, skiprows=[1])
    return data

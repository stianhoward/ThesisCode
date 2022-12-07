"""
tendeka_fit.py

Perform the curve fit of the 8-paramater estimation equation
Export the optimized parameters to a json(or pickle?) file
"""

import os.path as osp
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

sys.path.insert(1, '../../Thesis_Plots/')
import model_plot
import torch

CSV_TRAIN_PATH = "../../Thesis_Data/100-145TrainingData.csv"
CSV_VALID_PATH = "../../Thesis_Data/100-145ValidationData.csv"
CSV_TEST_PATH = "../../Thesis_Data/100-145TestData.csv"

RAND_CSV_PATH = "model_data"


def main():
    # Join the training and validation data since there's no distiction for this application
    train_data = load_data(CSV_TRAIN_PATH)
    valid_data = load_data(CSV_VALID_PATH)
    test_data = load_data(CSV_TEST_PATH)
    data = pd.concat([train_data, valid_data],ignore_index=True)

    # Optimize the function to the data
    opt_param = optimize(valid_data)

    # Export random values to extra data source for Transfer learning
    rand_data = generate_data(data.head(1), 10000, opt_param, data['Rate'].max())
    save_data(rand_data, RAND_CSV_PATH, [0.8, 0.1, 0.1])

    fig = model_plot.create_modelfigure(f, opt_param, data)

    plt.savefig("Predicted.jpg")
    plt.show()

    # Calculate error
    test_data['predictions'] = test_data.apply(lambda x: f(model_plot.retrieve_independent(x), *opt_param),axis=1)
    pred = torch.tensor(test_data['predictions'].values.tolist())
    target = torch.tensor(test_data['DPtotal'].values.tolist())
    loss = torch.nn.MSELoss()
    print(loss(pred, target).detach().tolist())


"""
load_data(csv_path)

Load the data to be fitted into the file, and return it as a pandas array
csv_path: path to the csv file to be loaded
"""
def load_data(csv_path):
    data = pd.read_csv(csv_path, skiprows=[1])
    return data

def save_data(data, dir_path, data_split):
    # Export full dataframe
    data.to_csv(osp.join(dir_path, 'RandComplete.csv'), index=False)

    #segregate data
    if data_split[0] != 0:
        # Training Data
        trainingData, data = partition_values(data, data_split[0])
        trainingData.to_csv(osp.join(dir_path, "RandTrainingData.csv"), index=False)
    if data_split[0] != 1:
        # There is leftover data for Validation/Testing
        if data_split[2] == 0:
            data.to_csv(osp.join(dir_path, "RandValidationData.csv"), index=False)
        elif data_split[1] == 0:
            data.to_csv(osp.join(dir_path, "RandTestData.csv"), index=False)
        else:
            frac = data_split[1] / (data_split[1] + data_split[2])
            validationData, testData = partition_values(data, frac)
            validationData.to_csv(osp.join(dir_path, "RandValidationData.csv"), index=False)
            testData.to_csv(osp.join(dir_path, "RandTestData.csv"), index=False)



def partition_values(dataframe, fraction):
    data = dataframe.sample(frac=fraction)
    remainder = dataframe.drop(data.index)
    return data, remainder

"""
optimize(data)

"""
def optimize(data):
    x_data = data[['Rate', 'Oil Fraction', 'rhoo', 'muo', 'Gas Fraction', 'rhog', 'mug', 'Water Fraction', 'rhow', 'muw']].T
    y_data = data['DPtotal']
    parameters, covariance = curve_fit(f, x_data.to_numpy(), y_data.to_numpy())
    return parameters


"""
f(ind_vars, rho_cal, mu_cal, mu_exp, a_aicd, flow_exp)

tendeka function to be optimized
ind_vars is sent as an array
ind_vars = [q, a_o, rho_o, mu_o, a_g, rho_g, mu_g, a_w, rho_w, mu_w]
parameters are single value inputs

currently rho_cal and mu_cal are assumed as 1 g/cc and 1 cp respectively
"""
#def f(ind_vars, rho_cal, mu_cal, mu_exp, a_aicd, flow_exp):
def f(ind_vars, mu_exp, a_aicd, flow_exp):
    rho_mix = ind_vars[1] * ind_vars[2] + ind_vars[4] * ind_vars[5] + ind_vars[7] * ind_vars[8]
    mu_mix = ind_vars[1] * ind_vars[3] + ind_vars[4] * ind_vars[6] + ind_vars[7] * ind_vars[9]
    f = rho_mix*rho_mix* (1/mu_mix)**mu_exp
    return f * a_aicd * ind_vars[0]**flow_exp


"""
generate_data(training_df, size, params)

Generate randomized data fitting 'f' with params for transfer learning

training_df: model data from which model is created, and constants are taken
size: number of randomized values to compute
params: parameters for tendeka_fit function
max_rate: maximum flow rate in the data set

return: dataframe to be exported and trained from
"""
def generate_data(training_df, size, params, max_rate):
    arr = np.array((size, training_df.shape[1]))
    rands = np.random.random((size, 3))
    sum_rands = np.sum(rands, axis=1)
    rands = rands / np.atleast_2d(sum_rands).T

    # BH Pressure column
    BHP = np.full((size, 1), training_df['B.H.Pressure'][0])
    aIndex = np.full((size, 1), training_df['Adiabatic Index'][0])
    aw = np.atleast_2d(rands[:,0]).T
    rhow = np.full((size, 1), training_df['rhow'][0])
    muw = np.full((size, 1), training_df['muw'][0])
    ao = np.atleast_2d(rands[:,1]).T
    rhoo = np.full((size, 1), training_df['rhoo'][0])
    muo = np.full((size, 1), training_df['muo'][0])
    ag = np.atleast_2d(rands[:,2]).T
    rhog = np.full((size, 1), training_df['rhog'][0])
    mug = np.full((size, 1), training_df['mug'][0])
    mixmu = muw * aw + muo * ao + mug * ag
    mixrho = rhow * aw + rhoo * ao + rhog * ag

    # Generate spread of flow rates
    rate = np.random.random((size, 1)) * max_rate

    arr = np.concatenate((rate, ao, rhoo, muo, ag, rhog, mug, aw, rhow, muw), axis = 1)

    #calculate Rates and DPtotal vectors
    dps = np.empty(size)
    for i in range(0,len(arr)):
        dps[i] = f(arr[i],params[0], params[1], params[2])
    dps = np.atleast_2d(dps).T


    data = np.concatenate((BHP, aIndex, rhow, muw, rhoo, muo, rhog, mug, rate, dps, aw, ao, ag, mixmu, mixrho), axis=1)

    return pd.DataFrame(data, columns = training_df.columns.tolist())


if __name__ == "__main__":
    main()

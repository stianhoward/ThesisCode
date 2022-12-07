"""
basicml.py

Basic Machine-learing functionality to train from real training data
"""

import os.path as osp

import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from pl_cross import Trainer
from pl_cross import TensorboardLogger

from model import Network

#device = "cude" if torch.cuda.is_available() else "cpu"
device = "cpu"

DATA_DIR="../../Thesis_Data"

RUN_NAME = "Test1"

def main(num_folds=5,model_path="models/"):
    # retrieve data into dataloader
    training_dl = getDataLoader(osp.join(DATA_DIR, "145TrainingData.csv"))
    validate_dl = getDataLoader(osp.join(DATA_DIR, "145ValidationData.csv"), shuffle=False)

    # Create and train the model
    model = Network()
    logger = TensorboardLogger(osp.join("lightning_logs", RUN_NAME))
    trainer = Trainer(
            num_folds = 5,
            shuffle=False,
            max_epochs=10,
            stratified=False,
            default_root_dir = osp.join(model_path,"chkpts"),
            logger = logger
            )
    # Need to figure out val_dl as well
    trainer.cross_validate(model, training_dl, validate_dl)

    ensemble_model = trainer.create_ensemble(model)

    # Save the model
    torch.save(ensemble_model.state_dict(), osp.join(model_path, RUN_NAME + "model.pt"))
    #model_scripted = torch.jit.script(ensemble_model)
    #model_scripted.save(osp.join(model_path, RUN_NAME + "model.pt"))
    #torch.save(ensemble_model, model_path + RUN_NAME + "model.pth")


    #torch.save(model.state_dict(), "models/" + RUN_NAME + "modelsave.pth")



def getDataLoader(path, shuffle=True):
    data = pd.read_csv(path, skiprows=[1])

    x_data = data[['Rate', 'Oil Fraction', 'rhoo', 'muo', 'Gas Fraction', 'rhog', 'mug', 'Water Fraction', 'rhow', 'muw']]
    y_data = data['DPtotal']

    out_data = []
    for x,y in zip(x_data.values.tolist(), y_data.values.tolist()):
        out_data.append([torch.tensor(x),torch.tensor([y])])

    dl = DataLoader(out_data, shuffle=shuffle, batch_size=5, num_workers=8)
    return dl


if __name__=="__main__":
    num_folds = 5
    main(num_folds)

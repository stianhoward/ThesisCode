"""
basicml.py

Basic Machine-learing functionality to train from real training data
"""

import os.path as osp
#from resource import getrusage, RUSAGE_SELF
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

#from pl_cross import Trainer
#from pl_cross import TensorboardLogger

from long_model import Network
import utils

#device = "cude" if torch.cuda.is_available() else "cpu"
device = "cpu"

DATA_DIR="../../Thesis_Data"

DATA_LOAD = "30" # 30 or 100
BHP = 145 # 145 or 150
RUN_NAME = f"basic-{DATA_LOAD}-{BHP}"

def main(num_folds=5,model_path="final/"):
    # retrieve data into dataloader
    training_dl = utils.getDataLoader(osp.join(DATA_DIR, f"{DATA_LOAD}-{BHP}TrainingData.csv"))
    validate_dl = utils.getDataLoader(osp.join(DATA_DIR, f"{DATA_LOAD}-{BHP}ValidationData.csv"), shuffle=False)
    test_dl = utils.getDataLoader(osp.join(DATA_DIR, f"{DATA_LOAD}-{BHP}TestData.csv"), shuffle=False)

    # Create and train the model
    model = Network()
    logger = pl.loggers.TensorBoardLogger(osp.join(model_path, "lightning_logs", RUN_NAME))
    trainer = pl.Trainer(
            max_epochs=200,
            default_root_dir = model_path,
            logger = logger,
            callbacks = [EarlyStopping(monitor="Validation loss", mode="min", patience=10)],
            auto_lr_find = True
            )

    # Actually train the model
    trainer.fit(model, training_dl, validate_dl)

    # Test the Model
    trainer.test(model, test_dl)

    # Save the model
    model_scripted = torch.jit.script(model)
    model_scripted.save(osp.join(model_path, RUN_NAME + "model.pt"))


    # Trace and export torchscript model
    example_data = utils.getOneValue(osp.join(DATA_DIR, f"{DATA_LOAD}-{BHP}TrainingData.csv"))
    traced_model = torch.jit.trace(model, example_data)
    traced_model.save(osp.join(model_path, RUN_NAME + "model_script.pt"))


if __name__=="__main__":
    num_folds = 5
    main(num_folds=num_folds)
    #print(getrusage(RUSAGE_SELF).ru_maxrss)

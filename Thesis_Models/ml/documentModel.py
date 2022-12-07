"""
Create and run a model, to get a performance, and finally plot results
"""

import os.path as osp
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

sys.path.insert(1, '../../Thesis_Plots/')
import model_plot
import utils
from model import Network


MODEL_PATH = "models/Regular1model.pt"
CSV_TRAIN_PATH = "../../Thesis_Data/145TrainingData.csv"
CSV_VALID_PATH = "../../Thesis_Data/145ValidationData.csv"
CSV_TEST_PATH = "../../Thesis_Data/145TestData.csv"


def main():
    model = torch.jit.load(MODEL_PATH)
    model.eval()

    #TODO: This should be abstracted out in some way
    # Join the training and validation data since there's no distiction for this application
    train_data = utils.load_pd_data(CSV_TRAIN_PATH)
    valid_data = utils.load_pd_data(CSV_VALID_PATH)
    data = pd.concat([train_data, valid_data],ignore_index=True)

    fig = model_plot.create_modelfigure(model, [], data, toTensor=True)
    plt.savefig("Pred-rename.jpg")
    plt.show()


def test():
    model = Network.load_from_checkpoint("models/lightning_logs/version_0/checkpoints/epoch=9-step=1799.ckpt")

    model.eval()

    test_dl = utils.getDataLoader(CSV_TEST_PATH, shuffle=False)
    trainer = pl.Trainer()
    trainer.test(model, dataloaders=test_dl)

if __name__ == "__main__":
    main()
    test()



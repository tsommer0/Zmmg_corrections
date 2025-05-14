#%%
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

from torchcfm.conditional_flow_matching import *
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.models.models import *
from torchcfm.utils import *

import argparse
import yaml
#%%
#custom imports

from ot_toy_generators import *
from ot_useful_functions import *
#%%

def load_config():
    parser = argparse.ArgumentParser(description='OT Toy Beispiel mit CFM bzw. Regression')
    parser.add_argument('--config',    type=str,   help='Pfad zur YAML-Konfigurationsdatei')
    parser.add_argument('--seed',      type=int,   default=42)
    parser.add_argument('--batch_size',type=int,   default=256)
    parser.add_argument('--n_epochs',  type=int,   default=15)
    parser.add_argument('--sigma',     type=float, default=0.01)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--n_data',    type=int,   default=500000)
    parser.add_argument('--train_fraction',type=float, default=0.6)
    parser.add_argument('--test_fraction', type=float, default=0.2)
    parser.add_argument('--val_fraction',  type=float, default=0.2)
    # zunächst nur --config parsen
    args, remaining = parser.parse_known_args()
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    return parser.parse_args()

#if wo nur dann nicht aktiviert wenn dieses py skript teil eines moduls ist
if __name__ == '__main__':
    args = load_config()
    #use cuda if available, else use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device = ", device)

#%%

#initialize the model
model = Regressor(input_dim=2).to(device)

#initialize loss and choose parameter optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#set the number of epochs and batch size
n_epochs = args.n_epochs
batch_size = args.batch_size


start = time.time()

#generate data
n_data = args.n_data
data = sample_8gaussians(n_data).to(device)
labels = sample_moons(n_data).to(device)

#split data
dataset = TensorDataset(data, labels)
train_size = int(args.train_fraction * len(dataset))
val_size = int(args.val_fraction * len(dataset))
test_size = int(args.test_fraction * len(dataset))
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

#prepare the dataloader function to generate batches
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

#prepare the dataloader function to generate batches
dataloader = DataLoader(
    TensorDataset(data, labels), batch_size=batch_size, shuffle=True,
)

#%%


array_loss_train = []
array_loss_val = []
for epoch in range(n_epochs):
    array_loss_train_epoch = []
    array_loss_val_epoch = []
    for step, (x0,x1) in enumerate(dataloader):
        optimizer.zero_grad()

        x0 = x0.to(device)
        x1 = x1.to(device)

        x1 = ot_matcher_caio(x0, x1)

        xm = model(x0)

        loss = loss_function(xm, x1)
        loss.backward()
        optimizer.step()
        array_loss_train_epoch.append(loss.item())

    test_size = 1024
    x0 = sample_8gaussians(test_size).to(device)
    x1 = sample_moons(test_size).to(device)
    morphed = model(x0)
    morph_comparison_plot(x0, x1, morphed, title=f"Morph Comparison Plot Epoch: {epoch}", filename=f"epoch_{epoch}_morph.png")

    model.eval()
    with torch.no_grad():
        for step, (x0, x1) in enumerate(val_loader):
            optimizer.zero_grad()

            #rearrange x1 array such that the entries are matched optimally to the x0 entries
            x1 = ot_matcher_caio(x0, x1)

            #creation of data label
            xm = model(x0)

            #determine loss
            loss = loss_function(xm, x1)

            #append loss in the loss array such that loss can be plotted later
            array_loss_val_epoch.append(loss.item())

    array_loss_train.append(np.mean(array_loss_train_epoch))
    array_loss_val.append(np.mean(array_loss_val_epoch))

lossplot_train(array_loss_train, array_loss_val, n_epochs, filename="loss_plot_botreg.png", out_dir="botreg_plots")
#%%
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
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

# Create an instance of the conditional flow matcher
sigma=args.sigma
cfm = ConditionalFlowMatcher(sigma=sigma)

#use mlp class from torchcfm.models expecting input of dim=dim and a time dimension
dim=2
model = MLP(dim=dim, time_varying=True).to(device)

#choose adam optimizer from torch and apply to parameters of model
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#chose MSE as loss function using MSELoss class from torch
loss_function = torch.nn.MSELoss()

#choose batch size, epochs
batch_size = args.batch_size
n_epochs = args.n_epochs


start = time.time()

#generate data
n_data = args.n_data
data = four_circles_spread(n_data).to(device)
labels = checker(n_data).to(device)


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


#%%


#train the model
array_loss_train = []
array_loss_val = []
for epoch in range(n_epochs):
    array_loss_train_epoch = []
    array_loss_val_epoch = []
    for step, (x0,x1) in enumerate(train_loader):
        optimizer.zero_grad()

        #rearrange x1 array such that the entries are matched optimally to the x0 entries
        x1 = ot_matcher_caio(x0, x1)

        #contiuous flow matching (creation of data label)
        t, xt, ut = cfm_t_xt_ut(x0, x1)

        #concatenate the sample location xt and t for the model input. For t from cfm_sample_location_and_conditional_flow:
        vt = model(torch.cat([xt, t], dim=-1))

        #determine loss
        loss = loss_function(vt, ut)

        #call backpropagated computation of gradient in network parameter space
        loss.backward()

        #call step to use the previously calculated gradient to update the model parameters according to the chosen optimizer
        optimizer.step()

        #append loss in the loss array such that loss can be plotted later
        array_loss_train_epoch.append(loss.item())


    #visualize the vector field transformation per n-iterations
    end = time.time()
    print(f"Epoch {epoch}: loss {loss.item():0.3f} time {(end - start):0.2f}")

    model.eval()
    with torch.no_grad():
        for step, (x0, x1) in enumerate(val_loader):
            optimizer.zero_grad()

            #rearrange x1 array such that the entries are matched optimally to the x0 entries
            x1 = ot_matcher_caio(x0, x1)

            #contiuous flow matching (creation of data label)
            t, xt, ut = cfm_t_xt_ut(x0, x1)

            #concatenate the sample location xt and t for the model input. For t from cfm_sample_location_and_conditional_flow:
            vt = model(torch.cat([xt, t], dim=-1))

            #determine loss
            loss = loss_function(vt, ut)

            #append loss in the loss array such that loss can be plotted later
            array_loss_val_epoch.append(loss.item())
    #visualize the vector field transformation per n-iterations

    start = end
    node = NeuralODE(
        torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    with torch.no_grad():
        traj = node.trajectory(
            checker(1024).to("cuda"),
            t_span=torch.linspace(0, 1, 10),
        )
        savefig_trajectories(traj.cpu().numpy())

    array_loss_train.append(np.mean(array_loss_train_epoch))
    array_loss_val.append(np.mean(array_loss_val_epoch))

    
# %%

lossplot_train(array_loss_train, array_loss_val, n_epochs)

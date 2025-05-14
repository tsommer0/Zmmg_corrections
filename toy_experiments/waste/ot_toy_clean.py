#%%
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

from torchcfm.conditional_flow_matching import *
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.models.models import *
from torchcfm.utils import *
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

#BOTCFM Cells

# Create an instance of the conditional flow matcher
sigma=.01
cfm = ConditionalFlowMatcher(sigma=sigma)

#use mlp class from torchcfm.models expecting input of dim=dim and a time dimension
dim=2
model = MLP(dim=dim, time_varying=True).to(device)

#choose adam optimizer from torch and apply to parameters of model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#chose MSE as loss function using MSELoss class from torch
loss_function = torch.nn.MSELoss()

#choose batch size, epochs
batch_size = 256
n_epochs = 5


start = time.time()

#generate data
n_data = 500000
data = four_circles_spread(n_data).to(device)
labels = checker(n_data).to(device)

#prepare the dataloader function to generate batches
dataloader = DataLoader(
    TensorDataset(data, labels), batch_size=batch_size, shuffle=True,
)

#%%
#choose batch size, epochs
batch_size = 256
n_epochs = 15


#train the model
array_loss = []
for epoch in range(n_epochs):
    for step, (x0,x1) in enumerate(dataloader):
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
        array_loss.append(loss.item())


    #visualize the vector field transformation per n-iterations
    end = time.time()
    print(f"Epoch {epoch}: loss {loss.item():0.3f} time {(end - start):0.2f}")
    start = end
    node = NeuralODE(
        torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    with torch.no_grad():
        traj = node.trajectory(
            checker(1024).to("cuda"),
            t_span=torch.linspace(0, 1, 10),
        )
        plot_trajectories(traj.cpu().numpy())
# %%

lossplot(array_loss, n_epochs)

# %%

import torch.nn as nn
#%%

#BOT + Discontinuous Morphing

#initialize the model
model = Regressor(input_dim=2).to(device)

#initialize loss and choose parameter optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


start = time.time()

#generate data
n_data = 500000
data = sample_8gaussians(n_data).to(device)
labels = sample_moons(n_data).to(device)

#prepare the dataloader function to generate batches
dataloader = DataLoader(
    TensorDataset(data, labels), batch_size=batch_size, shuffle=True,
)

#%%

#choose batch size, epochs
batch_size = 256
n_epochs = 5


array_loss = []
for epoch in range(n_epochs):
    for step, (x0,x1) in enumerate(dataloader):
        optimizer.zero_grad()

        x0 = x0.to(device)
        x1 = x1.to(device)

        x1 = ot_matcher_caio(x0, x1)

        xm = model(x0)

        loss = loss_function(xm, x1)
        loss.backward()
        optimizer.step()
        array_loss.append(loss.item())

    test_size = 1024
    x0 = sample_8gaussians(test_size).to(device)
    x1 = sample_moons(test_size).to(device)
    morphed = model(x0)
    morphed = morphed.cpu()
    x0 = x0.cpu()
    x1 = x1.cpu()

    plt.scatter(x0[:, 0], x0[:, 1], 
            c='blue', 
            alpha=0.5, 
            label='x0 (8 Gaussians)', 
            edgecolors='w', 
            s=30)

    # Plot x0 distribution
    plt.scatter(x1[:, 0], x1[:, 1], 
                c='green', 
                alpha=0.5, 
                label='x0 (8 Gaussians)', 
                edgecolors='w', 
                s=30)

    # Plot morphed distribution
    plt.scatter(morphed[:, 0].detach().numpy(), morphed[:, 1].detach().numpy(), 
                c='red', 
                alpha=0.5, 
                label='Morphed', 
                edgecolors='w', 
                s=30)
    
    plt.title(f"Scatter Plot of Original and Morphed Distributions, Epoch {epoch}")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#%%

lossplot(array_loss, n_epochs)
# %%

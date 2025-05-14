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

import sys
import argparse
import yaml

savedir = "~/models/8gaussian-moons"
os.makedirs(savedir, exist_ok=True)
#%%

#definition der config funktion, welche die config argumente parst
def load_config():
    parser = argparse.ArgumentParser(description='OT Toy Beispiel mit Config')
    parser.add_argument('--config',    type=str,   help='Pfad zur YAML-Konfigurationsdatei')
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
sigma = args.sigma

print(sigma)

a = bfiesg.esbfeuwsf

#%%
























#%%
#define functions to sample desired distributions
def circle(n_samples, r=1, sig=.025):
    phi_gen = np.random.uniform(0, 2*np.pi, size=n_samples)
    r_gen = np.random.normal(loc=r, scale=sig, size=n_samples)
    
    x = np.cos(phi_gen)*r_gen
    y = np.sin(phi_gen)*r_gen
    
    out = np.column_stack((x, y))
    
    return Tensor(out)

def four_circles_spread(n_samples, r=1, sig=.025, spread=.5):
    x = circle(n_samples, r, sig)

    n_circ = n_samples/4

    shift = spread*np.array([[1,1], [1,-1], [-1,1], [-1,-1]])

    for i in range(4):
        i_low = int(i*n_circ)
        i_high = int((i+1)*n_circ)
        x[i_low:i_high] = x[i_low:i_high] + shift[i]
        
    return x

def checker(n_samples, n_rows=4, n_columns=4, range_x=[-1,1], range_y=[-1,1], uneven_indices=1):
    
    start_x = range_x[0]
    start_y = range_y[0]
    
    if not (uneven_indices==0 or uneven_indices==1):
        raise ValueError("uneven_indices can only be 1 (represents True) or 0 (represents False)")
    
    range_x = (range_x[1]-range_x[0])/n_columns
    shift_x = np.random.uniform(0, range_x, size=n_samples)
    #column = np.random.randint(0, n_columns-1, size=n_samples)
    
    
    range_y = (range_y[1]-range_y[0])/n_rows
    shift_y = np.random.uniform(0, range_y, size=n_samples)
    #row = np.random.randint(0, n_rows-1, size=n_samples)
    
    cells = np.array([
    (i, j)
    for i in range(n_columns)
    for j in range(n_rows)
    if (i + j) % 2 == uneven_indices
    ])
    
    indices = np.random.choice(len(cells), size=n_samples, replace=True)
    cells = cells[indices]
    
    columns = cells[:,0]
    rows = cells[:,1]
    
    x = range_x*rows + shift_x + start_x
    y = range_y*columns + shift_y + start_y
    
    out = np.column_stack((x, y))
    
    return Tensor(out)

"""
def visualizer(n_samples, x0, x1, device, loss, steps=10, sigma=0.01):
    if (k + 1) % 50 == 0:
        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        start = end
        node = NeuralODE(
            torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )
        with torch.no_grad():
            traj = node.trajectory(
                checker(int(n_samples)).to(device),
                t_span=torch.linspace(0, 1, int(steps)),
            )
            plot_trajectories(traj.cpu().numpy())
"""


#self_written_optimal_transport_data_generator_using_pot
def ot_xt_ut(x0, x1, sigma=0.01):
    """
    Compute the optimal transport map and its gradient for two distributions.
    """
    #compute batch size and call the device the samples are stored on
    batch_size, device = x0.shape[0], x0.device

    #generate uniform weights for each point in the sample
    a=b=torch.ones(batch_size) / batch_size

    #compute cost matrix
    C = torch.cdist(x0, x1, p=2).pow(2)
    
    #compute optimal transport plan. for pot, all inputs need to be numpy arrays stored on the cpu
    gamma = pot.bregman.sinkhorn(a.cpu().numpy(), b.cpu().numpy(), C.cpu().numpy(), reg=sigma)
    gamma = torch.from_numpy(gamma).to(device)

    #transfer the weights back to the training device
    a = a.to(device)
    b = b.to(device)

    #compute barycenter
    barycenter = (gamma @ x1) / a[:, None]

    #compute vector field
    ut = barycenter - x0

    #compute torch tensor of vector field times
    t = torch.rand(batch_size, 1, device=device)

    #comute linearly interpolated points xt
    xt = x0 + t * (barycenter - x0)

    return t, xt, ut


#optimal transport plan generator using scipy
def ot_matcher_caio(x0, x1):
    with torch.no_grad():
        # Compute the cost matrix on CPU
        M = torch.cdist(x0, x1) ** 2  # Shape: (batch_size, batch_size)
        M_cpu = M.cpu().numpy()

        # Solve the linear sum assignment problem
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(M_cpu)

        # Reorder x1 based on the assignment
        x1_matched = x1[col_ind].to(device)

    return x1_matched

#%%

#use cuda if available, else use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#%%


# Create an instance of the conditional flow matcher
sigma=.01
cfm = ConditionalFlowMatcher(sigma=sigma)

#create an instance of the OTPlanSampler
ot_sampler = OTPlanSampler(method="sinkhorn", reg=sigma)

#use mlp class from torchcfm.models expecting input of dim=dim and a time dimension
dim=2
model = MLP(dim=dim, time_varying=True).to(device)

#choose adam optimizer from torch and apply to parameters of model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#chose MSE as loss function using MSELoss class from torch
loss_function = torch.nn.MSELoss()

batch_size = 256
n_epochs = 5
n_iterations = 3000
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

#train the model using BOTCFM
array_loss = []
for epoch in range(n_epochs):
    for step, (x0,x1) in enumerate(dataloader):
        optimizer.zero_grad()

        
        x1 = ot_matcher_caio(x0, x1)

        #x0 = checker(batch_size).to(device)
        #x1 = four_circles_spread(batch_size, .75, .0125, .25).to(device)

        #dataset from jupyter tutorial
        #x0 = sample_8gaussians(batch_size).to("cuda")
        #x1 = sample_moons(batch_size).to("cuda")

        # Use the ConditionalFlowMatcher to sample xt and compute the conditional flow (ut)
        #t, xt, ut = cfm.sample_location_and_conditional_flow(x0, x1)
        #t = t.unsqueeze(-1)

        #print(torch.cat([xt, t], dim=-1))

        #determine the optimal transport plan and compute the vector field using cfm library:
        #x0, x1 = ot_sampler.sample_plan(x0, x1)
        t = torch.rand(x0.shape[0], 1, device=device)
        xt = x0 + t * (x1 - x0)
        ut = x1 - x0

        #print(ut.shape)

        # Concatenate the sample location xt and t for the model input. For t from cfm_sample_location_and_conditional_flow:
        vt = model(torch.cat([xt, t], dim=-1))

        diff = ut - vt


        loss = loss_function(vt, ut)
        #loss = torch.mean((vt - ut) ** 2)

        loss.backward()
        optimizer.step()
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
            sample_8gaussians(1024).to("cuda"),
            t_span=torch.linspace(0, 1, 10),
        )
        plot_trajectories(traj.cpu().numpy())

#%%
plt.plot(np.linspace(0, n_epochs, len(array_loss)), array_loss, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()


#%%









#%%
#discontinuous model

import scipy
import torch.nn as nn
#%%

# Define the Regressor model
class Regressor(nn.Module):
    def __init__(self, input_dim=2):
        super(Regressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.model(x)

#initialize the model
model = Regressor(input_dim=2).to(device)

#initialize loss and choose parameter optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

batch_size = 256
n_epochs = 5
n_iterations = 3000
start = time.time()

#generate data
n_data = 500000
data = sample_8gaussians(n_data).to(device)
labels = sample_moons(n_data).to(device)

#prepare the dataloader function to generate batches
dataloader = DataLoader(
    TensorDataset(data, labels), batch_size=batch_size, shuffle=True,
)


def ot_matcher_caio(x0, x1):
    with torch.no_grad():
        # Compute the cost matrix on CPU
        M = torch.cdist(x0, x1) ** 2  # Shape: (batch_size, batch_size)
        M_cpu = M.cpu().numpy()

        # Solve the linear sum assignment problem
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(M_cpu)

        # Reorder x1 based on the assignment
        x1_matched = x1[col_ind].to(device)

    return x1_matched

#%%

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
plt.plot(np.linspace(0, n_epochs, len(array_loss)), array_loss, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()

# %%

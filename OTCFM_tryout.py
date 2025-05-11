#%%
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
from torch import Tensor
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

from torchcfm.conditional_flow_matching import *
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.models.models import *
from torchcfm.utils import *

savedir = "~/models/8gaussian-moons"
os.makedirs(savedir, exist_ok=True)
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
optimizer = torch.optim.Adam(model.parameters())

#chose MSE as loss function using MSELoss class from torch
loss_function = torch.nn.MSELoss()

batch_size = 256
start = time.time()
for k in range(30000):
    optimizer.zero_grad()

    #x0 = checker(batch_size).to(device)
    #x1 = four_circles_spread(batch_size, .75, .0125, .25).to(device)

    #dataset from jupyter tutorial
    x0 = sample_8gaussians(batch_size).to("cuda")
    x1 = sample_moons(batch_size).to("cuda")

    # Use the ConditionalFlowMatcher to sample xt and compute the conditional flow (ut)
    #t, xt, ut = cfm.sample_location_and_conditional_flow(x0, x1)
    t, xt, ut = ot_xt_ut(x0, x1, sigma=sigma)

    #determine the optimal transport plan and compute the vector field using cfm library:
    #x0, x1 = ot_sampler.sample_plan(x0, x1)
    #t = torch.rand(batch_size, 1, device=device)
    #xt = x0 + t * (x1 - x0)
    #ut = x1 - x0

    # Concatenate the sample location xt and t for the model input. For t from cfm_sample_location_and_conditional_flow:
    #vt = model(torch.cat([xt, t[:, None]], dim=-1))
    #Concatenate the sample location xt and t for the model input. For t from ot_xt_ut:
    vt = model(torch.cat([xt, t], dim=-1))
    loss = loss_function(vt, ut)
    #loss = torch.mean((vt - ut) ** 2)

    loss.backward()
    optimizer.step()

    #visualize the vector field transformation per n-iterations
    if (k + 1) % 500 == 0:
        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
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
    
# %%

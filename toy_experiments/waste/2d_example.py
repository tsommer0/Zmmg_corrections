#!/usr/bin/env python
# coding: utf-8

# In[17]:


import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *

savedir = "models/8gaussian-moons"
os.makedirs(savedir, exist_ok=True)


# # Block 1: Correcting with batchOT flow matching

# In[18]:


def sample_conditional_pt(x0, x1, t, sigma):
    """
    Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    xt : Tensor, shape (bs, *dim)

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    t = pad_t_like_x(t, x0)
    #t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    mu_t = t * x1 + (1 - t) * x0
    epsilon = torch.randn_like(x0)
    return mu_t + sigma * epsilon


# In[19]:


def compute_conditional_vector_field(x0, x1):
    """
    Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch

    Returns
    -------
    ut : conditional vector field ut(x1|x0) = x1 - x0

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    return x1 - x0


# In[20]:


sigma = 0.1
dim = 2
batch_size = 256
model = MLP(dim=dim, time_varying=True).to("cuda")
optimizer = torch.optim.Adam(model.parameters())

start = time.time()
for k in range(300):
    optimizer.zero_grad()

    x0 = sample_8gaussians(batch_size).to("cuda")
    x1 = sample_moons(batch_size).to("cuda")

    t = torch.rand(x0.shape[0]).type_as(x0)
    xt = sample_conditional_pt(x0, x1, t, sigma=0.01)
    ut = compute_conditional_vector_field(x0, x1)

    vt = model(torch.cat([xt, t[:, None]], dim=-1))
    loss = torch.mean((vt - ut) ** 2)

    loss.backward()
    optimizer.step()

    if (k + 1) % 50 == 0:
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



# In[21]:


outputson = traj.cpu().numpy()[-1]

test_size = 1024
x0 = sample_8gaussians(test_size)
x1 = sample_moons(test_size)

print( np.shape(x1) )
print( np.shape(outputson) )

plt.figure(figsize=(8, 6))

# Plot x0 distribution
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
plt.scatter(outputson[:, 0], outputson[:, 1], 
            c='red', 
            alpha=0.5, 
            label='Morphed', 
            edgecolors='w', 
            s=30)

plt.title('Scatter Plot of Original and Morphed Distributions')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# # Block 2: Correcting with Optimal transport + regression (Using neural networks)

# In[22]:


#### instead of flow matching we use batch OT + regression
import scipy
import torch.nn as nn

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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Initialize network, loss function, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Regressor().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

batch_size = 64
num_iterations = 30000

start = time.time()
for k in range(1, num_iterations + 1):
    model.train()
    optimizer.zero_grad()

    # Sample data
    x0 = sample_8gaussians(batch_size)
    x1 = sample_moons(batch_size)

    #### Batch Optimal Transport
    with torch.no_grad():
        # Compute the cost matrix on CPU
        M = torch.cdist(x0, x1) ** 2  # Shape: (batch_size, batch_size)
        M_cpu = M.cpu().numpy()

        # Solve the linear sum assignment problem
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(M_cpu)

        # Reorder x1 based on the assignment
        x1_matched = x1[col_ind].to(device)

    # Forward pass
    outputs = model(x0)

    # Compute loss
    loss = criterion(outputs, x1_matched)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Logging
    if k % 100 == 0 or k == 1:
        elapsed = time.time() - start
        print(f"Iteration {k}/{num_iterations} - Loss: {loss.item():.4f} - Time Elapsed: {elapsed:.2f}s")
        start = time.time()


# In[ ]:





# In[28]:


### evaluate the model ...
test_size = 1024
x0 = sample_8gaussians(test_size)
x1 = sample_moons(test_size)
morphed = model(x0)

print( morphed.size )
print( morphed[:10] )


# In[29]:


plt.figure(figsize=(8, 6))

# Plot x0 distribution
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

plt.title('Scatter Plot of Original and Morphed Distributions')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[30]:


plt.figure(figsize=(8, 6))

# Plot x0 distribution
plt.scatter(x1[:, 0], x1[:, 1], 
            c='blue', 
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

plt.title('Scatter Plot of Original and Morphed Distributions')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# # Block 3: Correcting with Optimal transport + regression (Using Boosted decision trees)

# In[26]:


##### Now we train a BDT
########################

import torch
import numpy as np
import scipy.optimize
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles, make_moons
from sklearn.multioutput import MultiOutputRegressor

def collect_data(batch_size, num_iterations):
    X = []
    Y = []
    start_time = time.time()
    
    for k in range(1, num_iterations + 1):
        # Sample data
        x0 = sample_8gaussians(batch_size)  # Shape: (batch_size, 2)
        x1 = sample_moons(batch_size)       # Shape: (batch_size, 2)
        
        #### Batch Optimal Transport
        # Compute the cost matrix
        M = np.linalg.norm(x0[:, np.newaxis, :] - x1[np.newaxis, :, :], axis=2) ** 2  # Shape: (batch_size, batch_size)
        
        # Solve the linear sum assignment problem
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(M)
        
        # Reorder x1 based on the assignment to get matched pairs
        x1_matched = x1[col_ind]  # Shape: (batch_size, 2)
        
        # Append to dataset
        X.append(x0)           # Inputs
        Y.append(x1_matched)   # Targets
        
        # Logging
        if k % 100 == 0 or k == 1:
            elapsed = time.time() - start_time
            print(f"Data Collection Iteration {k}/{num_iterations} - Time Elapsed: {elapsed:.2f}s")
            start_time = time.time()
    
    # Concatenate all batches
    X = np.vstack(X)  # Shape: (num_iterations * batch_size, 2)
    Y = np.vstack(Y)  # Shape: (num_iterations * batch_size, 2)
    
    return X, Y

def train_bdt(X, Y):
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Initialize the BDT model wrapped with MultiOutputRegressor
    base_regressor = GradientBoostingRegressor(
        n_estimators=150,        # Number of boosting stages
        learning_rate=0.1,       # Shrinkage rate
        max_depth=4,             # Maximum depth of the individual regression estimators
        random_state=42,
        loss='squared_error'     # Updated loss parameter
    )
    bdt_model = MultiOutputRegressor(base_regressor)
    
    # Train the model
    print("Training the BDT model...")
    start_time = time.time()
    bdt_model.fit(X_train, Y_train)
    training_time = time.time() - start_time
    print(f"BDT training completed in {training_time:.2f}s")
    
    # Evaluate the model
    Y_pred = bdt_model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    print(f"Test MSE: {mse:.4f}")
    
    return bdt_model, X_test, Y_test, Y_pred

def visualize_results(X_test, Y_test, Y_pred, num_points=1000):
    plt.figure(figsize=(8, 6))
    
    # Plot true mappings
    plt.scatter(Y_test[:num_points, 0], Y_test[:num_points, 1],
                c='green', alpha=0.5, label='True', edgecolors='w', s=20)
    plt.title('Target')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    
    # Plot predicted mappings
    plt.scatter(Y_pred[:num_points, 0], Y_pred[:num_points, 1],
                c='red', alpha=0.5, label='Predicted', edgecolors='w', s=20)
    plt.title('BDT Predicted Morphed Distribution')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    
    # Plot residuals
    plt.scatter(X_test[:num_points, 0], X_test[:num_points, 1],
                c='blue', alpha=0.5, label='Residuals', edgecolors='w', s=20)
    plt.title('Nominal')
    plt.xlabel('Residual Feature 1')
    plt.ylabel('Residual Feature 2')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# In[27]:


def main():
    # Parameters
    batch_size = 256
    num_iterations = 2000  # Adjust based on available memory and computational resources
    
    # Step 1: Data Collection
    print("Starting data collection...")
    X, Y = collect_data(batch_size, num_iterations)
    print(f"Data collection completed. Total samples: {X.shape[0]}")
    
    # Step 2: Train BDT Model
    bdt_model, X_test, Y_test, Y_pred = train_bdt(X, Y)
    
    # Step 3: Visualize Results
    visualize_results(X_test, Y_test, Y_pred, num_points=1024)
    
if __name__ == "__main__":
    main()


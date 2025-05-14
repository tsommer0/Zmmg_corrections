#%%
import torch
import torch.nn as nn
import numpy as np
import scipy.optimize
import os
import matplotlib.pyplot as plt
#%%


#optimal transport plan generator using scipy
def ot_matcher_caio(x0, x1):
    with torch.no_grad():
        # Compute the cost matrix on CPU
        M = torch.cdist(x0, x1) ** 2  # Shape: (batch_size, batch_size)
        M_cpu = M.cpu().numpy()

        # Solve the linear sum assignment problem
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(M_cpu)

        # Reorder x1 based on the assignment
        x1_matched = x1[col_ind].to(x0.device)

    return x1_matched

#%%

#generate the data and labels for the CFM from the base and target samples
def cfm_t_xt_ut(x0, x1):
    """
    Function to compute t, xt, and ut for the conditional flow matcher.
    :param x0: Input tensor of shape (batch_size, dim)
    :param x1: Target tensor of shape (batch_size, dim)
    :return: t, xt, ut
    """
    # Compute t
    t = torch.rand(x0.shape[0], 1).to(x0.device)

    # Compute xt and ut
    xt = x0 + t * (x1 - x0)
    ut = x1 - x0

    return t, xt, ut

def lossplot_train(array_loss_train, array_loss_val, n_epochs, filename="loss_plot_botcfm.png", out_dir="botcfm_plots"):
    """
    Function to plot the loss over epochs.
    :param array_loss: List of loss values
    :param n_epochs: Number of epochs
    """

    plt.plot(np.linspace(0, n_epochs, n_epochs), array_loss_train, label='Training loss')
    plt.plot(np.linspace(0, n_epochs, n_epochs), array_loss_val, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.abspath(os.path.join(out_dir, filename))
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

#%%

#define the neural network model for the discontiuous NF morphing
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
    
#%%

#plot_trajectories function from cfm with savefig
def savefig_trajectories(traj, filename="final_epoch_trajectory_plot.png", out_dir="botcfm_plots"):
    """Plot trajectories of some selected samples and save the figure."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.abspath(os.path.join(out_dir, filename))
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

def morph_comparison_plot(x0, x1, morphed, title="Morph Comparison Plot", filename="morph_comparison_plot.png", out_dir="botreg_plots"):
    """
    Function to plot the original and morphed distributions.
    :param x0: Original distribution
    :param x1: Target distribution
    :param morphed: Morphed distribution
    """
    plt.figure(figsize=(8, 6))
    x0 = x0.cpu()
    x1 = x1.cpu()
    morphed = morphed.cpu()

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
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.show()
    plt.close()
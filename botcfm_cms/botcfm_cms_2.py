#%%
import math
import os
import time
import pandas as pd

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

from torchdiffeq import odeint

from torchcfm.conditional_flow_matching import *
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.models.models import *
from torchcfm.utils import *

import sys
import argparse
import yaml

#%%
#custom imports
from bsc_sommer_project_module.bot.balanced import ot_matcher_caio
from bsc_sommer_project_module.general.datahandling import generate_tensors, unstandardize_tensor, random_splits
from bsc_sommer_project_module.general.structuring import apply_ot_strategy, lossplot_train
from bsc_sommer_project_module.cfm import cfm_t_xt_ut, cfm_morphing

#%%
#load config

def load_config():
    parser = argparse.ArgumentParser(description='OT Toy Beispiel mit CFM bzw. Regression')
    parser.add_argument('--config',    type=str,   help='Pfad zur YAML-Konfigurationsdatei')
    parser.add_argument('--event_location',      type=str,   default='all')
    parser.add_argument('--batch_size',type=int,   default=2048)
    parser.add_argument('--n_epochs',  type=int,   default=40)
    parser.add_argument('--sigma',     type=float, default=0.001)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--train_fraction',type=float, default=0.6)
    parser.add_argument('--val_fraction',  type=float, default=0.2)
    parser.add_argument('--gpu_abuse', type=int, default=4, help='Number of workers for DataLoader')

    parser.add_argument('--OT_strategy', type=str, default='None', choices=['None', 'caio_OT', 'hard_UOT'], help='OT strategy to use')

    parser.add_argument('--variables_cms', type=str, default=[
            "photon_r9", 
            "photon_sieie",
            "photon_etaWidth",
            "photon_phiWidth",
            "photon_sieip",
            "photon_s4"
            ])
    parser.add_argument('--variables_mc', type=str, default=[
            "photon_raw_r9", 
            "photon_raw_sieie",
            "photon_raw_etaWidth",
            "photon_raw_phiWidth",
            "photon_raw_sieip",
            "photon_raw_s4"
            ])
    parser.add_argument('--conditions', type=str, default=[
            "photon_pt",
            "photon_ScEta",
            "photon_phi",
            "Rho_fixedGridRhoAll",
            "photon_muon_near_dR"
            ])
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
#initialize the variable vectors
variables_cms_names = args.variables_cms
variables_mc_names = args.variables_mc
conditions_names = args.conditions

#initialize config variables for the script
num_workers = args.gpu_abuse
batch_size = args.batch_size
n_epochs = args.n_epochs
lr = args.lr
sigma=args.sigma
ot_strategy = args.OT_strategy
print("Using the following OT strategy: ", ot_strategy)

train_fraction = args.train_fraction
val_fraction = args.val_fraction
#%%
#load the data as pandas dataframes
data_df = pd.read_pickle('../data_read_and_store/data_df.pkl')
mc_df = pd.read_pickle('../data_read_and_store/mc_df.pkl')

#%%

#load the data as tensors
target, standardized_target, mean_target, std_target = generate_tensors(data_df, variables_cms_names)
base, standardized_base, mean_base, std_base = generate_tensors(mc_df, variables_mc_names)
conditions, standardized_conditions, mean_conditions, std_conditions = generate_tensors(mc_df, conditions_names)

#splitting the data in training, validation and test set
target_train_tensor, target_val_tensor, target_test_tensor = random_splits(standardized_target, train_fraction, val_fraction)
base_train_tensor, base_val_tensor, base_test_tensor = random_splits(standardized_base, train_fraction, val_fraction)
conditions_train_tensor, conditions_val_tensor, conditions_test_tensor = random_splits(standardized_conditions, train_fraction, val_fraction)
#%%

# DataLoaders for MC
base_train_loader = DataLoader(base_train_tensor, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
base_val_loader   = DataLoader(base_val_tensor,   batch_size=batch_size, num_workers=num_workers,shuffle=False, drop_last=True)
base_test_loader  = DataLoader(base_test_tensor,  batch_size=batch_size, num_workers=num_workers,shuffle=False)

# DataLoaders for data
target_train_loader = DataLoader(target_train_tensor, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
target_val_loader   = DataLoader(target_val_tensor,   batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
target_test_loader  = DataLoader(target_test_tensor,  batch_size=batch_size, num_workers=num_workers, shuffle=False)

# DataLoaders for conditions
conditions_train_loader = DataLoader(conditions_train_tensor, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
conditions_val_loader   = DataLoader(conditions_val_tensor,   batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
conditions_test_loader  = DataLoader(conditions_test_tensor,  batch_size=batch_size, num_workers=num_workers, shuffle=False)

#%%

# Create an instance of the conditional flow matcher
cfm = ConditionalFlowMatcher(sigma=sigma)

#initialize the vector field model
model = MLP(dim=len(variables_mc_names+conditions_names), out_dim=len(variables_cms_names), time_varying=True).to(device)

#define the loss function
loss_function = torch.nn.MSELoss()

#choose adam optimizer from torch and apply to parameters of model
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#%%
#initialize the loss arrays
array_loss_train = []
array_loss_val = []
mse_morph = []


#start the training loop
start = time.time()

for epoch in range(n_epochs):
    array_loss_train_epoch = []
    array_loss_val_epoch = []
    mse_morph_epoch = []
    for step, (base_batch, conditions_batch, target_batch) in enumerate(zip(base_train_loader, conditions_train_loader, target_train_loader)):
        optimizer.zero_grad()
        base_batch = base_batch.to(device)
        conditions_batch = conditions_batch.to(device)
        target_batch = target_batch.to(device)

        # apply the selected OT strategy (ot_strategy string variable from the config) by rearranging the batches
        base_batch, conditions_batch, target_batch = apply_ot_strategy(base_batch, conditions_batch, target_batch, ot_strategy)

        #calculate the optimal transport field direction and magnitude ut and its position xt at time t from conditional flow matching
        t, xt, ut = cfm.sample_location_and_conditional_flow(base_batch, target_batch)

        #concatenate the sample location xt, the conditions_batch and t for the model input
        model_input = torch.cat([xt, conditions_batch, t.unsqueeze(-1)], dim=1)
        #note: this fixates the model input to always be a concat of (xt, condition, t)

        #calculate the model output vt
        vt = model(model_input)

        #determine loss
        loss = loss_function(vt, ut)

        #call backpropagated computation of gradient in network parameter space
        loss.backward()

        # apply gradient clipping 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        #call step to use the previously calculated gradient to update the model parameters according to the chosen optimizer
        optimizer.step()

        #append loss in the loss array such that loss can be plotted later
        array_loss_train_epoch.append(loss.item())

    print("Epoch: ", epoch)
    print("Mean loss: ", np.mean(array_loss_train_epoch))

    model.eval()
    with torch.no_grad():
        for step, (base_batch, conditions_batch, target_batch) in enumerate(zip(base_val_loader, conditions_val_loader, target_val_loader)):
            optimizer.zero_grad()
            base_batch = base_batch.to(device)
            conditions_batch = conditions_batch.to(device)
            target_batch = target_batch.to(device)

            # apply the selected OT strategy
            base_batch, conditions_batch, target_batch = apply_ot_strategy(base_batch, conditions_batch, target_batch, args.OT_strategy)

            #calculate the optimal transport field direction and magnitude ut and its position xt at time t from conditional flow matching
            t, xt, ut = cfm.sample_location_and_conditional_flow(base_batch, target_batch)
            
            #concatenate the sample location xt, the conditions_batch and t for the model input
            model_input = torch.cat([xt, conditions_batch, t.unsqueeze(-1)], dim=1)
            #note: this fixates the model input to always be a concat of (xt, condition, t)

            #calculate the model output vt
            vt = model(model_input)

            #determine loss
            loss = loss_function(vt, ut)

            #append loss in the loss array such that loss can be plotted later
            array_loss_val_epoch.append(loss.item())
            '''
            #calculate MSE between x(1) as calculated by integrating v(t) and target_batch to see if the model is able to learn the target distribution
            morphed_batch = cfm_morphing(model, base_batch, conditions_batch)

            mse = loss_function(morphed_batch, target_batch)
            mse_morph_epoch.append(mse.item())
            '''

        
        end = time.time()
        print(f"Epoch {epoch} completed in {(end-start):0.2f} seconds. ")
        start = end

        array_loss_train.append(np.mean(array_loss_train_epoch))
        array_loss_val.append(np.mean(array_loss_val_epoch))
        #mse_morph.append(np.mean(mse_morph_epoch)) 


# Save the trained model state dict for later use
model_save_path = "botcfm_cms_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model state dict saved to {model_save_path}")

lossplot_train(array_loss_train, array_loss_val, n_epochs, filename="loss_plot_botcfm_cms.png", out_dir="botcfm_plots")
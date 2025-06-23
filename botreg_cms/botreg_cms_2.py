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

from torchcfm.conditional_flow_matching import *
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.models.models import *
from torchcfm.utils import *

import sys
import argparse
import yaml

from bsc_sommer_project_module.general.datahandling import generate_tensors
#%%
#custom imports
from botreg_useful_functions import *
sys.path.append(os.path.abspath("../../toy_experiments/final_toy_networks"))
from ot_useful_functions import *

#%%
#load config

def load_config():
    parser = argparse.ArgumentParser(description='OT Toy Beispiel mit CFM bzw. Regression')
    parser.add_argument('--config',    type=str,   help='Pfad zur YAML-Konfigurationsdatei')
    parser.add_argument('--event_location',      type=str,   default='all')
    parser.add_argument('--batch_size',type=int,   default=512)
    parser.add_argument('--n_epochs',  type=int,   default=40)
    parser.add_argument('--sigma',     type=float, default=0.001)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--train_fraction',type=float, default=0.6)
    parser.add_argument('--val_fraction',  type=float, default=0.2)

    parser.add_argument('--OT_strategy', type=str, default='caio_OT', choices=['None', 'caio_OT', 'hard_UOT'], help='OT strategy to use')

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


def apply_ot_strategy(base_batch, conditions_batch, target_batch, strategy):
    """
    Dispatches to the appropriate OT matching strategy based on `strategy`.
    """
    if strategy == 'hard_UOT':
        return uot_matching(base_batch, conditions_batch, target_batch)
    elif strategy == "caio_OT":
        return base_batch, conditions_batch, ot_matcher_caio(base_batch, target_batch)
    elif strategy == 'None':
        # no transport; return batches unchanged
        return base_batch, conditions_batch, target_batch
    else:
        raise ValueError(f"Invalid OT strategy: {strategy}. Choose 'None' or 'hard_UOT'.")





#%%
#initialize the variable vectors

variables_cms = args.variables_cms
variables_mc = args.variables_mc
conditions_names = args.conditions

#initialize config variables for the script
batch_size = args.batch_size
n_epochs = args.n_epochs
lr = args.lr

#%%
#load the data as pandas dataframes

data_df = pd.read_pickle('../data_read_and_store/data_df.pkl')
mc_df = pd.read_pickle('../data_read_and_store/mc_df.pkl')

#%%
#differenciate between barrel, endcap and all events

eta_regions     = ['barrel', 'endcap']
data_eta_masks  = [
    np.abs(data_df["photon_ScEta"].values) < 4.442,
    np.abs(data_df["photon_ScEta"].values) > 1.566
]
mc_eta_masks    = [
    np.abs(mc_df["photon_ScEta"].values) < 4.442,
    np.abs(mc_df["photon_ScEta"].values) > 1.566
]

if args.event_location == 'all':
    use_data_df = data_df
    use_mc_df = mc_df
elif args.event_location == 'barrel':
    use_data_df = data_df[data_eta_masks[0]]
    use_mc_df = mc_df[mc_eta_masks[0]]
elif args.event_location == 'endcap':
    use_data_df = data_df[data_eta_masks[1]]
    use_mc_df = mc_df[mc_eta_masks[1]]
else:
    raise ValueError("Invalid event location. Choose 'all', 'barrel', or 'endcap'.")

#%%

#convert dataframe to tensor. Notation: base-Distribution = MC-Distribution, target-Distribution = CMS-Distribution
target = torch.from_numpy(use_data_df[variables_cms].to_numpy()).float()
base = torch.from_numpy(use_mc_df[variables_mc].to_numpy()).float()
conditions = torch.from_numpy(use_data_df[conditions_names].to_numpy()).float()

#calculate the mean and standard deviation of the data
target, standardized_target, mean_target, std_target = generate_tensors(use_data_df, variables_cms)
base, standardized_base, mean_base, std_base = generate_tensors(use_mc_df, variables_mc)
conditions, standardized_conditions, mean_conditions, std_conditions = generate_tensors(use_mc_df, conditions_names)


#redefine the unstandardized tensors as the standardized ones, such that the script does not have to be changed from then on
target = standardized_target
base = standardized_base
conditions = standardized_conditions


# random splits for base tensor
n_mc = len(base)
perm_mc = torch.randperm(n_mc)
train_mc_end = int(args.train_fraction * n_mc)
val_mc_end = train_mc_end + int(args.val_fraction * n_mc)
base_train_tensor = base[perm_mc[:train_mc_end]]
base_val_tensor   = base[perm_mc[train_mc_end:val_mc_end]]
base_test_tensor  = base[perm_mc[val_mc_end:]]

# random splits for target tensor
n_data = len(target)
perm_data = torch.randperm(n_data)
train_data_end = int(args.train_fraction * n_data)
val_data_end = train_data_end + int(args.val_fraction * n_data)
target_train_tensor = target[perm_data[:train_data_end]]
target_val_tensor   = target[perm_data[train_data_end:val_data_end]]
target_test_tensor  = target[perm_data[val_data_end:]]

# random splits for conditions tensor
n_conditions = len(conditions)
perm_conditions = torch.randperm(n_conditions)
train_conditions_end = int(args.train_fraction * n_conditions)
val_conditions_end = train_conditions_end + int(args.val_fraction * n_conditions)
conditions_train_tensor = conditions[perm_conditions[:train_conditions_end]]
conditions_val_tensor   = conditions[perm_conditions[train_conditions_end:val_conditions_end]]
conditions_test_tensor  = conditions[perm_conditions[val_conditions_end:]]



# DataLoaders for MC
base_train_loader = DataLoader(base_train_tensor, batch_size=batch_size, shuffle=False, drop_last=True)
base_val_loader   = DataLoader(base_val_tensor,   batch_size=batch_size, shuffle=False, drop_last=True)
base_test_loader  = DataLoader(base_test_tensor,  batch_size=batch_size, shuffle=False)

# DataLoaders for data
target_train_loader = DataLoader(target_train_tensor, batch_size=batch_size, shuffle=True, drop_last=True)
target_val_loader   = DataLoader(target_val_tensor,   batch_size=batch_size, shuffle=False, drop_last=True)
target_test_loader  = DataLoader(target_test_tensor,  batch_size=batch_size, shuffle=False)

# DataLoaders for conditions
conditions_train_loader = DataLoader(conditions_train_tensor, batch_size=batch_size, shuffle=False, drop_last=True)
conditions_val_loader   = DataLoader(conditions_val_tensor,   batch_size=batch_size, shuffle=False, drop_last=True)
conditions_test_loader  = DataLoader(conditions_test_tensor,  batch_size=batch_size, shuffle=False)
#%%
#initialize the model

model = Regressor2(input_dim=len(variables_mc+conditions_names), output_dim=len(variables_cms)).to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

M_cpu_buf = np.empty((batch_size, batch_size), dtype=np.float64)

array_loss_train = []
array_loss_val = []
for epoch in range(n_epochs):
    array_loss_train_epoch = []
    array_loss_val_epoch = []
    for step, (base_batch, conditions_batch, target_batch) in enumerate(zip(base_train_loader, conditions_train_loader, target_train_loader)):
        optimizer.zero_grad()
        base_batch = base_batch.to(device)
        conditions_batch = conditions_batch.to(device)
        target_batch = target_batch.to(device)

        



        # apply the selected OT strategy
        base_batch, conditions_batch, target_batch = apply_ot_strategy(
            base_batch, conditions_batch, target_batch, args.OT_strategy
        )

        base_batch_expansion = torch.cat([base_batch, conditions_batch], dim=1)

        morphed_base_batch = model(base_batch_expansion)

        loss = loss_function(morphed_base_batch, target_batch)

        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
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
            base_batch, conditions_batch, target_batch = apply_ot_strategy(
                base_batch, conditions_batch, target_batch, args.OT_strategy
            )

            #creation of data label
            base_batch_expansion = torch.cat([base_batch, conditions_batch], dim=1)

            morphed_base_batch = model(base_batch_expansion)

            loss = loss_function(morphed_base_batch, target_batch)

            #append loss in the loss array such that loss can be plotted later
            array_loss_val_epoch.append(loss.item())

    
    array_loss_train.append(np.mean(array_loss_train_epoch))
    array_loss_val.append(np.mean(array_loss_val_epoch))

lossplot_train(array_loss_train, array_loss_val, n_epochs, filename="loss_plot_botreg_cms_alpha.png", out_dir="botreg_plots")

# Save the trained model state dict for later use
model_save_path = "botreg_cms_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model state dict saved to {model_save_path}")
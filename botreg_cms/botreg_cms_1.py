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
#%%
#custom imports
sys.path.append(os.path.abspath("../../toy_experiments/final_toy_networks"))
from ot_useful_functions import *


#sys.path.append(os.path.abspath("../../Zmmg_samples_validation"))
#from plot_validation import perform_reweighting
#%%

def load_config():
    parser = argparse.ArgumentParser(description='OT Toy Beispiel mit CFM bzw. Regression')
    parser.add_argument('--config',    type=str,   help='Pfad zur YAML-Konfigurationsdatei')
    parser.add_argument('--event_location',      type=str,   default='all')
    parser.add_argument('--batch_size',type=int,   default=256)
    parser.add_argument('--n_epochs',  type=int,   default=20)
    parser.add_argument('--sigma',     type=float, default=0.01)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--n_data',    type=int,   default=500000)
    parser.add_argument('--train_fraction',type=float, default=0.6)
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


data_df = pd.read_pickle('../data_read_and_store/data_df.pkl')
mc_df = pd.read_pickle('../data_read_and_store/mc_df.pkl')



#%%
#IGNORE


variables = [
    # barrel
    "dimuon_mass"                            # invariant mass for both muons as calculated by the measured 4 momentum
    ,"mmy_mass"                              # invariant mass for all decay particles as calculated by the measured 4 momentum
    ,"mmy_pt"                                # transverse momentum of all decay particles
    ,"photon_energyRaw"                      # photon energy deposit in the ECAL
    ,"photon_muon_near_dR"                   # dR between photon and nearest muon
    ,"photon_mvalD_raw"                      # photon ID calculated by CMS BDT with values -1, ..., +1 where +1 = photon. Called 'MVA-discriminator'
    ,"photon_phi"                            # photon emission direction detector coordinate
    ,"photon_pt"                             # transverse momentum of the photon
    ,"photon_raw_ecalPFClusterIso"           # isolation variable: sum of neutral particle-cluster energies from particle flow algorithm within a cone around the photon in the ECAL
    ,"photon_raw_energyErr"                  # uncertainty on the ECAL photon energy deposit
    ,"photon_raw_esEffSigmaRR"               # width of the preshower deposit before the ECAL endcap
    ,"photon_raw_esEnergyOverRawE"           # ratio of preshower energy to raw supercluster endcap energy
    ,"photon_raw_etaWidth"                   # width of the endcap supercluster energy deposit distribution in eta
    ,"photon_raw_hcalPFClusterIso"           # isolation variable: sum of PF-cluster energy within a cone around the photon in the ECAL
    ,"photon_raw_hoe"                        # 'Hcal Over Ecal' ratio of energy deposits in both calorimeters
    ,"photon_raw_pfChargedIsoWorstVtx"       # largest isolation variable of photons using the pt of particle flow candidates (particles) associated with the photon vertex within a dR cone around that photon
    ,"photon_raw_phiWidth"                   # width of the endcap supercluster energy deposit distribution in phi
    ,"photon_raw_r9"                         # fraction of 3x3 energy deposit over total energy deposit in the ECAL supercluster measurement array
    ,"photon_raw_s4"                         # fraction of 2x2 energy deposit over 5x5 energy deposit in the ECAL supercluster measurement array
    ,"photon_raw_sieie"                      # uncertainty of energy distribution in eta dimension
    ,"photon_raw_sieip"                      # correlation of energy distribution in eta and phi
    ,"photon_raw_trkSumPtHollowConeDR03"     # transverse momentum activity within a hollow cone around the photon (.02<dR<.03)
    ,"photon_raw_trkSumPtSolidConeDR04"      # transverse momentum activity within a solid cone around the photon (0<=dR<.04)
    ,"photon_ScEta"                          # photon emission direction detector coordinate
    ,"Rho_fixedGridRhoAll"                   # ??? 'pileup' energy density in eta-phi-space from extra proton energy deposits ???correct???
]

variables = [
          "photon_energyRaw",
          "photon_pt",
          "photon_r9",
          "photon_sieie",
          "photon_etaWidth",
          "photon_phiWidth",
          "photon_sieip",
          "photon_s4",
          "photon_hoe",
          "photon_ecalPFClusterIso",
          "photon_trkSumPtHollowConeDR03",
          "photon_trkSumPtSolidConeDR04",
          "photon_pfChargedIsoWorstVtx",
          "photon_esEffSigmaRR",
          "photon_esEnergyOverRawE",
          "photon_hcalPFClusterIso",
          "photon_mvaID",
          "photon_ScEta",
          "photon_energyErr",
          "photon_phi",
          "Rho_fixedGridRhoAll",
          "mmy_mass",
          "mmy_pt",
          "dimuon_mass",
          "photon_muon_near_dR"
        ]


#%%

#DO NOT IGNORE

variables_data = [
            "photon_r9", 
            "photon_sieie",
            "photon_etaWidth",
            "photon_phiWidth",
            "photon_sieip",
            "photon_s4"
            ]

variables_mc = [
            "photon_raw_r9", 
            "photon_raw_sieie",
            "photon_raw_etaWidth",
            "photon_raw_phiWidth",
            "photon_raw_sieip",
            "photon_raw_s4"
            ]

conditions = [
            "photon_pt",
            "photon_ScEta",
            "photon_phi",
            "Rho_fixedGridRhoAll"
            ]


#%%
batch_size = args.batch_size
n_epochs = args.n_epochs
lr = args.lr

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

#convert dataframe to tensor
use_data_tensor = torch.from_numpy(use_data_df[variables_data].to_numpy()).float()
use_mc_tensor = torch.from_numpy(use_mc_df[variables_mc+conditions].to_numpy()).float()


# random splits for MC tensor
n_mc = len(use_mc_tensor)
perm_mc = torch.randperm(n_mc)
train_mc_end = int(args.train_fraction * n_mc)
val_mc_end = train_mc_end + int(args.val_fraction * n_mc)
mc_train_tensor = use_mc_tensor[perm_mc[:train_mc_end]]
mc_val_tensor   = use_mc_tensor[perm_mc[train_mc_end:val_mc_end]]
mc_test_tensor  = use_mc_tensor[perm_mc[val_mc_end:]]

# random splits for data tensor
n_data = len(use_data_tensor)
perm_data = torch.randperm(n_data)
train_data_end = int(args.train_fraction * n_data)
val_data_end = train_data_end + int(args.val_fraction * n_data)
data_train_tensor = use_data_tensor[perm_data[:train_data_end]]
data_val_tensor   = use_data_tensor[perm_data[train_data_end:val_data_end]]
data_test_tensor  = use_data_tensor[perm_data[val_data_end:]]

# DataLoaders for MC
train_mc_loader = DataLoader(mc_train_tensor, batch_size=batch_size, shuffle=True, drop_last=True)
val_mc_loader   = DataLoader(mc_val_tensor,   batch_size=batch_size, shuffle=False, drop_last=True)
test_mc_loader  = DataLoader(mc_test_tensor,  batch_size=batch_size, shuffle=False)

# DataLoaders for data
train_data_loader = DataLoader(data_train_tensor, batch_size=batch_size, shuffle=True, drop_last=True)
val_data_loader   = DataLoader(data_val_tensor,   batch_size=batch_size, shuffle=False, drop_last=True)
test_data_loader  = DataLoader(data_test_tensor,  batch_size=batch_size, shuffle=False)

#%%

def matcher_unbalanced(x0, x1):
    """
    Function to match the data and MC samples using unbalanced optimal transport.
    :param x0: Input tensor of shape (batch_size, dim)
    :param x1: Target tensor of shape (batch_size, dim)
    :return: Matched tensor
    """
    # Compute the cost matrix
    C = torch.cdist(x0, x1, p=2)

    # Compute the unbalanced optimal transport plan
    G = pot.unbalanced.sinkhorn_unbalanced(a, b, M, reg, reg_m_kl)

    # Apply the transport plan to the target tensor
    matched_x1 = torch.matmul(G, x1)

    return matched_x1




model = Regressor(input_dim=len(variables_mc+conditions), output_dim=len(variables_data)).to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

array_loss_train = []
array_loss_val = []
for epoch in range(n_epochs):
    array_loss_train_epoch = []
    array_loss_val_epoch = []
    for step, (x0_mc_batch, x1_data_batch) in enumerate(zip(train_mc_loader, train_data_loader)):
        optimizer.zero_grad()
        x0 = x0_mc_batch.to(device)
        x1 = x1_data_batch.to(device)

        #x1 = ot_matcher_caio(x0, x1)

        xm = model(x0)

        loss = loss_function(xm, x1)

        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        array_loss_train_epoch.append(loss.item())

    print("Epoch: ", epoch)
    print("Mean loss: ", np.mean(array_loss_train_epoch))

    model.eval()
    with torch.no_grad():
        for step, (x0_mc_batch, x1_data_batch) in enumerate(zip(val_mc_loader, val_data_loader)):
            optimizer.zero_grad()
            x0 = x0_mc_batch.to(device)
            x1 = x1_data_batch.to(device)

            #rearrange x1 array such that the entries are matched optimally to the x0 entries
            #x1 = ot_matcher_caio(x0, x1)

            #creation of data label
            xm = model(x0)

            #determine loss
            loss = loss_function(xm, x1)

            #append loss in the loss array such that loss can be plotted later
            array_loss_val_epoch.append(loss.item())

    
    array_loss_train.append(np.mean(array_loss_train_epoch))
    array_loss_val.append(np.mean(array_loss_val_epoch))

lossplot_train(array_loss_train, array_loss_val, n_epochs, filename="loss_plot_botreg_cms_alpha.png", out_dir="botreg_plots")


#%%
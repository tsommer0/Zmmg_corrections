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
from bsc_sommer_project_module.general.datahandling import generate_tensors, unstandardize_tensor, random_splits, event_location_filter
from bsc_sommer_project_module.general.structuring import apply_ot_strategy, initialize_model, load_weights, apply_morphing_strategy, lossplot_train
from bsc_sommer_project_module.cfm import cfm_t_xt_ut, cfm_morphing

#%%
#custom imports from plotting scripts, for the copied plotting parts
sys.path.append(os.path.abspath("../../Zmmg_samples_validation"))
from plot_validation import *
from flow_plotting_general_script import *
import zmmg_mvaID_eff as eff
import selection_utils

#%%

def load_config():
    parser = argparse.ArgumentParser(description='OT Toy Beispiel mit CFM bzw. Regression')
    parser.add_argument('--config',    type=str,   help='Pfad zur YAML-Konfigurationsdatei')
    parser.add_argument('--event_location',      type=str,   default='all', choices=['all', 'barrel', 'endcap'], help='Event location to use for the analysis')
    parser.add_argument('--morphing_strategy', type=str, default='regular', choices=['cfm', 'regular'], help='Morphing strategy to use')
    parser.add_argument('--path', type=str, default='../general_training_scripts/model_weights/regular.pth', help='Path to the model weights file')

    parser.add_argument('--variables_cms_names', type=str, default=[
            "photon_r9", 
            "photon_sieie",
            "photon_etaWidth",
            "photon_phiWidth",
            "photon_sieip",
            "photon_s4",
            "photon_energyErr"
            ])
    parser.add_argument('--variables_mc_names', type=str, default=[
            "photon_raw_r9", 
            "photon_raw_sieie",
            "photon_raw_etaWidth",
            "photon_raw_phiWidth",
            "photon_raw_sieip",
            "photon_raw_s4",
            "photon_raw_energyErr"
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
#initialize the variable column name vectors
variables_cms_names = args.variables_cms_names
variables_mc_names = args.variables_mc_names
conditions_names = args.conditions
path = args.path


#initialize further variables
morphing_strategy = args.morphing_strategy
print(f"Using morphing strategy: {morphing_strategy} with the weights from {path}")

#initialize the model
model = initialize_model(
    base_dimensions=variables_mc_names,
    conditions_dimensions=conditions_names,
    target_dimensions=variables_cms_names,
    strategy=morphing_strategy,
    device=device
)
#load the model weights
load_weights(model, device, path)

#%%
#load the data
data_df = pd.read_pickle('../data_read_and_store/data_df.pkl')
mc_df = pd.read_pickle('../data_read_and_store/mc_df.pkl')

#%%
#differenciate between barrel, endcap and all events
event_location = args.event_location
print(f"Using event location: {event_location}")
use_data_df, use_mc_df, eta_regions, data_eta_masks, mc_eta_masks = event_location_filter(data_df, mc_df, event_location)

#%%
#generating tensors and the standardization from the DataFrames
target, standardized_target, mean_target, std_target = generate_tensors(use_data_df, variables_cms_names)
base, standardized_base, mean_base, std_base = generate_tensors(use_mc_df, variables_mc_names)
conditions, standardized_conditions, mean_conditions, std_conditions = generate_tensors(use_mc_df, conditions_names)

#apply the chosen morphing to the base (=mc) data
morphed_base = apply_morphing_strategy(
    model=model,
    base_batch=standardized_base.to(device),
    conditions_batch=standardized_conditions.to(device),
    strategy=morphing_strategy
)

#revert standardization
morphed_base = unstandardize_tensor(morphed_base, mean_base.to(device), std_base.to(device))
# Detach and convert to numpy for plotting
morphed_base = morphed_base.detach().cpu().numpy()

#build morphed DataFrame with "<variable>_morphed" column names
morphed_cols = [f"{var}_morphed" for var in variables_mc_names]
morphed_df = pd.DataFrame(morphed_base, columns=morphed_cols, index=use_mc_df.index)

#after here: copied parts from the plotting script

'''
result_df = use_data_df[variables_cms_names].join(
    use_mc_df[variables_mc_names+conditions_names]
)
result_df = result_df.join(morphed_df)

print(len(result_df.columns))

'''


# --- rename the arrays according to caio plotting script ---
var_list = variables_mc_names
var_list_corr = [f"{entry}_morphed" for entry in variables_mc_names]
data_var_list = variables_cms_names

year = "2024"
total_lumi = "35.9 fb^{-1}" #idk where this comes from
outdir = f"./plots_{year}/"
os.makedirs(outdir, exist_ok=True)

data_df = use_data_df[variables_cms_names]
# Include weights column in mc_df so it can be used below
mc_df = use_mc_df[variables_mc_names + ["weights"]]
mc_df = mc_df.join(morphed_df)

for entry, entry_corr, entry_data in zip(var_list, var_list_corr, data_var_list):
        if len(data_df) == 0 or len(mc_df) == 0:
            continue

        # get full-data mean/std for dynamic bin edges
        mean = np.nanmean(data_df[entry_data].values)
        std  = np.nanstd( data_df[entry_data].values )
        std  = std if std > 0 else 0.1

        # --- exactly your 28-bin logic from the first script ---
        if 'Iso' in entry:
            xbins = hist.new.Reg(30, 0.0, mean + 4.5*std, overflow=True)
        elif 'es' in entry:
            xbins = hist.new.Reg(30, 0.0, mean + 2.8*std, overflow=True)
        elif 'hoe' in entry:
            xbins = hist.new.Reg(30, 0.0, 0.08, overflow=True)
        elif 'DR' in entry:
            xbins = hist.new.Reg(30, 0.0, 4.0, overflow=True)
        elif 'energy' in entry:
            xbins = hist.new.Reg(30, 0.0, mean + 3.0*std, overflow=True)
        elif 'pt' in entry or 'Rho' in entry:
            xbins = hist.new.Reg(30, mean - 4*std, mean + 4*std, overflow=True)
        elif 'eta' in entry and 'idth' not in entry:
            xbins = hist.new.Reg(30, -2.5, 2.5, overflow=True)
        elif 'sieie' in entry:
            xbins = hist.new.Reg(30, mean - 3.0*std, mean + 3.0*std, overflow=True)
        elif 'sieip' in entry:
            xbins = hist.new.Reg(30, mean - 3.0*std, mean + 3.0*std, overflow=True)
        elif 'r9' in entry:
            xbins = hist.new.Reg(30, 0.5, mean + 1.5*std, overflow=True)
        elif 'mva' in entry:
            xbins = hist.new.Reg(30, -1.0, 1.0, overflow=True)
        elif 's4' in entry:
            xbins = hist.new.Reg(30, mean - 4.5*std, mean + 2.5*std, overflow=True)
        elif 'dimuon_mass' in entry:
            xbins = hist.new.Reg(42, 35, 80, overflow=True)
        else:
            xbins = hist.new.Reg(30, mean - 2.0*std, mean + 3.0*std, overflow=True)
        # ------------------------------------------------------------

        # now fill & save for each η-region
        for region, dmask, mmask in zip(eta_regions, data_eta_masks, mc_eta_masks):
            endcap = (region == 'endcap')

            data_vals    = data_df[entry_data].values[dmask]
            mc_vals      = mc_df[entry].values[mmask]
            mc_corr_vals = mc_df[entry_corr].values[mmask]
            weights      = len(data_vals) * mc_df["weights"][mmask].values

            data_hist    = xbins.Weight()
            mc_hist      = xbins.Weight()
            mc_corr_hist = xbins.Weight()

            data_hist.fill(    data_vals)
            mc_hist.fill(      mc_vals,      weight=weights)
            mc_corr_hist.fill( mc_corr_vals, weight=weights)

            out_name = os.path.join(outdir, f"{region}_{entry}.pdf")
            plot_utils.plott(
                data_hist, mc_hist, mc_corr_hist,
                out_name,
                physics_process = r'$Z\rightarrow \mu \mu \gamma$',
                lumi_label      = total_lumi,
                zmmg            = True,
                xlabel          = str(entry),
                postEE          = True,
                endcap          = endcap
            )
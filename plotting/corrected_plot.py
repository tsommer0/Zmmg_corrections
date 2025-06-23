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
from bsc_sommer_project_module.general.structuring import apply_ot_strategy, initialize_model, load_weights, apply_morphing_strategy_plot, lossplot_train
from bsc_sommer_project_module.cfm import cfm_t_xt_ut, cfm_morphing_plot

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
    parser.add_argument('--batch_size',type=int,   default=128)
    parser.add_argument('--n_epochs',  type=int,   default=50)
    parser.add_argument('--sigma',     type=float, default=0.001)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--hidden_dimensions', type=int, default=128, help='Number of hidden dimensions in the model')
    #TO DO : Implement layers:
    parser.add_argument('--layers', type=int, default=5, help='Number of layers in the model')

    parser.add_argument('--OT_strategy', type=str, default='caio_OT', choices=['None', 'caio_OT', 'hard_UOT'], help='OT strategy used')
    parser.add_argument('--morphing_strategy', type=str, default='cfm', choices=['cfm', 'regular'], help='Morphing strategy used')
    parser.add_argument('--best_model', type=bool, default=True, help='Use the best model or the nth epoch model')

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

#initialize further variables
morphing_strategy = args.morphing_strategy
hidden_dimensions = args.hidden_dimensions
ot_strategy = args.OT_strategy
layers = args.layers
lr = args.lr
batch_size = args.batch_size
epochs = args.n_epochs
best_model = args.best_model

#preparing the directory for saving the training results
#-----------UNNECESSARY HERE. IS COMBINED WITH PLOTS{year} BELOW IN CAIOS PLOTTING CODE----------------
#out_dir = f'./corrected_plot_1/{morphing_strategy}_{ot_strategy}/{layers}_layers/{hidden_dimensions}_hidden_dimensions/{lr}_learning_rate/{batch_size}_batch_size'
#os.makedirs(out_dir, exist_ok=True)

if best_model == False:
    path_to_weights = f'../general_training_scripts/training_results_1/{morphing_strategy}_{ot_strategy}/{layers}_layers/{hidden_dimensions}_hidden_dimensions/{lr}_learning_rate/{batch_size}_batch_size/epochs{epochs}.pth'
    print(f"Using morphing strategy: {morphing_strategy} with the weights from {path_to_weights}")
elif best_model == True:
    path_to_weights = f'../general_training_scripts/training_results_1/{morphing_strategy}_{ot_strategy}/{layers}_layers/{hidden_dimensions}_hidden_dimensions/{lr}_learning_rate/{batch_size}_batch_size/best_model.pth'
    print(f"Using morphing strategy: {morphing_strategy} with the weights from {path_to_weights}")

#initialize the model
model = initialize_model(
    base_dimensions=variables_mc_names,
    conditions_dimensions=conditions_names,
    target_dimensions=variables_cms_names,
    layers=layers,
    hidden_dimensions=hidden_dimensions,
    strategy=morphing_strategy,
    device=device
)
#load the model weights
load_weights(model, device, path_to_weights)

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

# Perform batched morphing to avoid CUDA OOM
model.eval()
plot_batch_size = 1024  # adjust this value if needed
morphed_chunks = []
with torch.no_grad():
    total_samples = standardized_base.size(0)
    for start_idx in range(0, total_samples, plot_batch_size):
        end_idx = min(start_idx + plot_batch_size, total_samples)
        base_batch_i = standardized_base[start_idx:end_idx].to(device)
        cond_batch_i = standardized_conditions[start_idx:end_idx].to(device)
        morphed_batch_i = apply_morphing_strategy_plot(
            model=model,
            base_batch=base_batch_i,
            conditions_batch=cond_batch_i,
            strategy=morphing_strategy
        )
        morphed_batch_i = unstandardize_tensor(morphed_batch_i, mean_base.to(device), std_base.to(device))
        morphed_chunks.append(morphed_batch_i.detach().cpu().numpy())
morphed_base = np.concatenate(morphed_chunks, axis=0)

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
if best_model==False:
    outdir = f'./plots_{year}/{morphing_strategy}_{ot_strategy}/{layers}_layers/{hidden_dimensions}_hidden_dimensions/{lr}_learning_rate/{batch_size}_batch_size/{epochs}_epochs'
elif best_model==True:
    outdir = f'./plots_{year}/{morphing_strategy}_{ot_strategy}/{layers}_layers/{hidden_dimensions}_hidden_dimensions/{lr}_learning_rate/{batch_size}_batch_size/best_model'
os.makedirs(outdir, exist_ok=True)

'''
# Standardize use_data_df entirely (no weights here)
use_data_df = (use_data_df - use_data_df.mean()) / use_data_df.std()

# Preserve weights from MC before standardizing feature columns
weights = use_mc_df["weights"]

# Standardize only the MC feature columns, not weights
use_mc_df[variables_mc_names] = (use_mc_df[variables_mc_names] - use_mc_df[variables_mc_names].mean()) / use_mc_df[variables_mc_names].std()

# Restore the original weights column
use_mc_df["weights"] = weights'''

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



print(f"Plots saved to {outdir}")
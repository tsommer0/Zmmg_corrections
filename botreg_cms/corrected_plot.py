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

from botreg_useful_functions import *
#%%
#custom imports
sys.path.append(os.path.abspath("../../Zmmg_samples_validation"))
from plot_validation import *
from flow_plotting_general_script import *
import zmmg_mvaID_eff as eff
import selection_utils

#%%

def load_config():
    parser = argparse.ArgumentParser(description='OT Toy Beispiel mit CFM bzw. Regression')
    parser.add_argument('--config',    type=str,   help='Pfad zur YAML-Konfigurationsdatei')
    parser.add_argument('--event_location',      type=str,   default='all')
    parser.add_argument('--batch_size',type=int,   default=512)
    parser.add_argument('--n_epochs',  type=int,   default=20)
    parser.add_argument('--sigma',     type=float, default=0.01)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--train_fraction',type=float, default=0.6)
    parser.add_argument('--val_fraction',  type=float, default=0.2)

    parser.add_argument('--OT_strategy', type=str, default='caio_OT', choices=['None', 'caio_OT', 'hard_UOT'], help='OT strategy to use')

    parser.add_argument('--variables_cms_names', type=str, default=[
            "photon_r9", 
            "photon_sieie",
            "photon_etaWidth",
            "photon_phiWidth",
            "photon_sieip",
            "photon_s4"
            ])
    parser.add_argument('--variables_mc_names', type=str, default=[
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
#initialize the variable column name vectors
variables_cms_names = args.variables_cms_names
variables_mc_names = args.variables_mc_names
conditions_names = args.conditions

#%%
#load the data

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

target, standardized_target, mean_target, std_target = generate_tensors(use_data_df, variables_cms_names)
base, standardized_base, mean_base, std_base = generate_tensors(use_mc_df, variables_mc_names)
conditions, standardized_conditions, mean_conditions, std_conditions = generate_tensors(use_mc_df, conditions_names)


base_expansion = torch.cat([standardized_base, standardized_conditions], dim=1).to(device)

# Re-create model architecture
model = Regressor2(input_dim=len(variables_mc_names+conditions_names), output_dim=len(variables_cms_names)).to(device)
#model = MLP(dim=len(variables_mc_names+conditions_names), out_dim=len(variables_cms_names), time_varying=True).to(base.device)
# Load saved parameters, mapping to the correct device and using weights_only mode
state_dict = torch.load("botreg_cms_model.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()


#for the cfm morphing
'''
# define ODE function for conditioned vector field
def ode_func(t, x):
    # x: [batch_size, 1, input_dim]
    x_flat = x.squeeze(1)
    # time tensor of shape [batch_size, 1]
    t_tensor = t * torch.ones(x_flat.size(0), 1, device=x_flat.device)
    # concatenate [x, conditions, t]
    inp = torch.cat([x_flat, conditions, t_tensor], dim=1)
    # compute vector field
    v = model(inp)
    # return with same batch/sequence shape
    return v.unsqueeze(1)

# integrate only endpoints from t=0 to t=1
t_span = torch.tensor([0.0, 1.0], device=base.device)
traj = odeint(
    ode_func,
    base.unsqueeze(1),
    t_span,
    atol=1e-4,
    rtol=1e-4,
    method="dopri5",
)
# extract x(1) and remove the singleton dimension
morphed_base = traj[-1].squeeze(1)
'''

# for the discrete morphing:
morphed_base = model(base_expansion)

morphed_base = unstandardize_tensor(morphed_base, mean_base.to(device), std_base.to(device))

morphed_base = morphed_base.detach().cpu().numpy()

# Build morphed DataFrame with "<variable>_morphed" column names
morphed_cols = [f"{var}_morphed" for var in variables_mc_names]
morphed_df = pd.DataFrame(morphed_base, columns=morphed_cols, index=use_mc_df.index)

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
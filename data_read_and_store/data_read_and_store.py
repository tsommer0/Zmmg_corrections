#%%
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ot as pot
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split
#import torchdyn
#from torchdyn.core import NeuralODE
#from torchdyn.datasets import generate_moons

#from torchcfm.conditional_flow_matching import *
#from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
#from torchcfm.models.models import *
#from torchcfm.utils import *

#%%
#custom imports
sys.path.append(os.path.abspath("../../toy_experiments/final_toy_networks"))
#from ot_toy_generators import *
from ot_useful_functions import *

sys.path.append(os.path.abspath("../../Zmmg_samples_validation"))
from plot_validation import *
import zmmg_mvaID_eff as eff
import selection_utils
#%%
config_path = sys.argv[1]
config = load_config(config_path)

# 1. Load the JSON config
with open(config_path, "r") as f:
    config = json.load(f)

# 1. Load config, data, and MC
config, data_df, mc_df, total_lumi = read_data(config)
year      = config.get("year", "unknown")
mvaID_cut = config.get("mvaID_cut", 0.2)
Do_flows_validation = config.get("Do_flows_validation", False)

### variable lists to read
var_list_corr   = config["variables"]["var_list_corr"]
data_var_list   = config["variables"]["data_var_list"]
var_list        = config["variables"]["var_list"]
conditions_list = config["variables"]["conditions_list"]

### removing nan values from the dataframe
var_list_data = [ var.replace('_raw', '') for var in var_list ]
data_df = data_df.dropna(subset=var_list_data + conditions_list)
mc_df = mc_df.dropna(subset=var_list + var_list_data + conditions_list)
#%%

# get the directory this script resides in
script_dir = os.path.dirname(os.path.abspath(__file__))

# use that as the default output_dir
output_dir = config.get("output_dir", script_dir)
os.makedirs(output_dir, exist_ok=True)
data_df.to_pickle(os.path.join(output_dir, "data_df_pure_new.pkl"))
mc_df.to_pickle(os.path.join(output_dir, "mc_df_pure_new.pkl"))
print(f"DataFrames saved to {output_dir}/data_df_pure_new.pkl and mc_df_pure_new.pkl")



# Apply random scaling: each entry is multiplied by (1 + epsilon),
# where epsilon is drawn from a standard normal distribution.
# If desired, you can control the scale of the perturbation by setting a variable epsilon_std.
epsilon_std = 0.01  # standard deviation for epsilon

# For data_df
eps_data = np.random.normal(loc=0.0, scale=epsilon_std, size=data_df.shape)
eps_data_df = pd.DataFrame(eps_data, index=data_df.index, columns=data_df.columns)
data_df = data_df * (1 + eps_data_df)

# For mc_df
eps_mc = np.random.normal(loc=0.0, scale=epsilon_std, size=mc_df.shape)
eps_mc_df = pd.DataFrame(eps_mc, index=mc_df.index, columns=mc_df.columns)
mc_df = mc_df * (1 + eps_mc_df)

data_df.to_pickle(os.path.join(output_dir, "data_df_smeared_new.pkl"))
mc_df.to_pickle(os.path.join(output_dir, "mc_df_smeared_new.pkl"))
print(f"Smeared DataFrames saved to {output_dir}/data_df_smeared_new.pkl and mc_df_smeared_new.pkl")

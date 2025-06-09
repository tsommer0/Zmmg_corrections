import pickle
import sys
import os
import traceback
import torch
import numpy as np
import pandas as pd


from bsc_sommer_project_module.general.structuring import initialize_model, apply_morphing_strategy
from bsc_sommer_project_module.general.datahandling import event_location_filter, generate_tensors, unstandardize_tensor


sys.path.append(os.path.abspath("../../Zmmg_samples_validation"))
from plot_validation import *
from flow_plotting_general_script import *




#----------------------------------------------------------------------unpickle the payload received from mother script 3.py----------------------------------------------------------------------
print("child: about to unpickle payload", file=sys.stderr)
try:
    payload = pickle.load(sys.stdin.buffer)
    print("child: got keys", list(payload.keys()), file=sys.stderr)
    # now look up each key exactly as you’ve written:
    variables_mc_names = payload["variables_mc_names"]
    conditions_names   = payload["conditions_names"]
    variables_cms_names= payload["variables_cms_names"]
    data_test_df       = payload["data_test_df"]
    mc_test_df         = payload["mc_test_df"]
    eta_regions        = payload["eta_regions"]
    data_eta_masks     = payload["data_eta_masks"]
    mc_eta_masks       = payload["mc_eta_masks"]
    variables_cms_names= payload["variables_cms_names"]
    variables_mc_names = payload["variables_mc_names"]
    conditions_names   = payload["conditions_names"]
    morphing_strategy  = payload["morphing_strategy"]
    hidden_dimensions  = payload["hidden_dimensions"]
    ot_strategy        = payload["ot_strategy"]
    layers             = payload["layers"]
    lr                 = payload["lr"]
    batch_size         = payload["batch_size"]
    device             = payload["device"]
    event_location     = payload["event_location"]
    sigma_cfm          = payload["sigma_cfm"]
    year               = payload["year"]
except Exception as e:
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

print("child: built dataframes successfully", file=sys.stderr)


#----------------------------------------------------------------------initialize the model and load weights----------------------------------------------------------------------

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

#load weights into the model
if morphing_strategy == 'cfm':
    path_to_weights = f'../general_training_scripts/training_results_3/{morphing_strategy}_{ot_strategy}/{sigma_cfm}_sigma_cfm/{layers}_layers/{hidden_dimensions}_hidden_dimensions/{lr}_learning_rate/{batch_size}_batch_size/best_model.pth'
elif morphing_strategy == 'regular':
    path_to_weights = f'../general_training_scripts/training_results_3/{morphing_strategy}_{ot_strategy}/{layers}_layers/{hidden_dimensions}_hidden_dimensions/{lr}_learning_rate/{batch_size}_batch_size/best_model.pth'
state_dict = torch.load(path_to_weights, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

print(f"Using morphing strategy: {morphing_strategy} with the weights from {path_to_weights}")




#----------------------------------------------------------------------prepare the data and apply model----------------------------------------------------------------------

use_data_df, use_mc_df, eta_regions, data_eta_masks, mc_eta_masks = event_location_filter(data_test_df, mc_test_df, event_location)

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
        morphed_batch_i = apply_morphing_strategy(
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

# --- rename the arrays according to caio plotting script ---
var_list = variables_mc_names
var_list_corr = [f"{entry}_morphed" for entry in variables_mc_names]
data_var_list = variables_cms_names


#after here: copied parts from the plotting script-----------------------------------------------------------------------
#----------------------------------------------------------------------plotting the results----------------------------------------------------------------------


total_lumi = "35.9 fb^{-1}" #idk where this comes from
if morphing_strategy == 'cfm':
    outdir = f'../plotting/plots_{year}/{morphing_strategy}_{ot_strategy}/{sigma_cfm}_sigma_cfm/{layers}_layers/{hidden_dimensions}_hidden_dimensions/{lr}_learning_rate/{batch_size}_batch_size/best_model'
elif morphing_strategy == 'regular':
    outdir = f'../plotting/plots_{year}/{morphing_strategy}_{ot_strategy}/{layers}_layers/{hidden_dimensions}_hidden_dimensions/{lr}_learning_rate/{batch_size}_batch_size/best_model'
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

print(f"Plots saved to {outdir}")
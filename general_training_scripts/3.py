import argparse
import yaml
import os
import sys
import time
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
import pandas as pd
import numpy as np
import pickle
import subprocess
from pathlib import Path



from bsc_sommer_project_module.general.structuring import initialize_model, apply_ot_strategy, lossplot_train
from bsc_sommer_project_module.general.datahandling import event_location_filter, generate_tensors, random_splits, loss_function


#----------------------------------------------------------------------set the seeds----------------------------------------------------------------------
# Enable deterministic cuBLAS algorithms for cdist reproducibility
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

generator = torch.Generator().manual_seed(42)

torch.use_deterministic_algorithms(True)  # Ensures reproducibility in PyTorch

#----------------------------------------------------------------------load config----------------------------------------------------------------------



def load_config():
    parser = argparse.ArgumentParser(description='OT Toy Beispiel mit CFM bzw. Regression')
    parser.add_argument('--config',    type=str,   help='Pfad zur YAML-Konfigurationsdatei')
    parser.add_argument('--year',      type=str,   default='2024_2', help='Year of the data')
    parser.add_argument('--event_location',      type=str,   default='all')
    parser.add_argument('--batch_size',type=int,   default=8196)
    parser.add_argument('--n_epochs',  type=int,   default=1000)
    parser.add_argument('--scheduler_patience', type=int, default=10, help='Patience for the learning rate scheduler')
    parser.add_argument('--sigma',     type=float, default=0.5)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--train_fraction',type=float, default=0.7)
    parser.add_argument('--val_fraction',  type=float, default=0.1)
    parser.add_argument('--gpu_abuse', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--save_progress', type=bool, default=False, help='True enables saving the model.pth and loss plot if epoch+1 % 10 == 0')
    parser.add_argument('--hidden_dimensions', type=int, default=512, help='Number of hidden dimensions in the model')
    #TO DO : Implement layers:
    parser.add_argument('--layers', type=int, default=4, help='Number of layers in the model')

    parser.add_argument('--OT_strategy', type=str, default='hard_UOT', choices=['None', 'caio_OT', 'hard_UOT'], help='OT strategy to use')
    parser.add_argument('--morphing_strategy', type=str, default='cfm', choices=['cfm', 'regular'], help='Morphing strategy to use')

    parser.add_argument('--variables_cms', type=str, default=[
            "photon_r9", 
            "photon_sieie",
            "photon_etaWidth",
            "photon_phiWidth",
            "photon_sieip",
            "photon_s4",
            "photon_energyErr"
            ])
    parser.add_argument('--variables_mc', type=str, default=[
            "photon_raw_r9", 
            "photon_raw_sieie",
            "photon_raw_etaWidth",
            "photon_raw_phiWidth",
            "photon_raw_sieip",
            "photon_raw_s4",
            "photon_raw_energyErr",
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



#--------------------------------------------------------------------------------------------------------------------------------------------
#initialize variables from the config
variables_cms_names = args.variables_cms
variables_mc_names = args.variables_mc
conditions_names = args.conditions
year = args.year
event_location = args.event_location
num_workers = args.gpu_abuse
batch_size = args.batch_size
n_epochs = args.n_epochs
lr = args.lr
sigma_cfm=args.sigma
ot_strategy = args.OT_strategy
morphing_strategy = args.morphing_strategy
layers = args.layers
hidden_dimensions = args.hidden_dimensions
scheduler_patience = args.scheduler_patience
early_stop_patience = scheduler_patience*2+1
save_progress = args.save_progress
train_fraction = args.train_fraction
val_fraction = args.val_fraction
print(f"Using morphing strategy: {morphing_strategy} with the OT strategy {ot_strategy}")
#create output directory for the training result
if morphing_strategy == 'cfm':
    out_dir_models = f'./training_results_3/{morphing_strategy}_{ot_strategy}/{sigma_cfm}_sigma_cfm/{layers}_layers/{hidden_dimensions}_hidden_dimensions/{lr}_learning_rate/{batch_size}_batch_size'
elif morphing_strategy == 'regular':
    out_dir_models = f'./training_results_3/{morphing_strategy}_{ot_strategy}/{layers}_layers/{hidden_dimensions}_hidden_dimensions/{lr}_learning_rate/{batch_size}_batch_size'

os.makedirs(out_dir_models, exist_ok=True)



#----------------------------------------------------------------------initialize model, optimizer and scheduler----------------------------------------------------------------------
#initialize variables for the training

model = initialize_model(
    base_dimensions=variables_mc_names,
    conditions_dimensions=conditions_names,
    target_dimensions=variables_cms_names,
    layers=layers,
    hidden_dimensions=hidden_dimensions,
    strategy=morphing_strategy,
    device=device
)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# Learning rate scheduler: reduce LR on plateau of validation loss
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=scheduler_patience,
)

array_loss_train = []
array_loss_val = []


#----------------------------------------------------------------------load the data----------------------------------------------------------------------

#load the dataframes
try:
    data_df = pd.read_pickle('../data_read_and_store/data_df_smeared.pkl')
    mc_df = pd.read_pickle('../data_read_and_store/mc_df_smeared.pkl')
except Exception as e:
    print("Error loading dataframes (Hint: Script uses relative paths. So only start the script from ~/Zmmg_corrections/general_plotting_scripts):", e)
    sys.exit(1)

#filter the dataframes by the event location
data_df, mc_df, eta_regions, data_eta_masks, mc_eta_masks = event_location_filter(data_df, mc_df, event_location)

#load the event weights to insert into the loss function
weights, standardized_weights, mean_weights, std_weights = generate_tensors(mc_df, ['weights'])
#take absolute of weights and normalize for numerical stability
weights = weights.abs()
weights = weights / weights.mean().item()

#convert the dataframes to tensors
target, standardized_target, mean_target, std_target = generate_tensors(data_df, variables_cms_names)
base, standardized_base, mean_base, std_base = generate_tensors(mc_df, variables_mc_names)
conditions, standardized_conditions, mean_conditions, std_conditions = generate_tensors(mc_df, conditions_names)

#standardize target usiing base mean and base std
standardized_target = (target - mean_base) / std_base

#split the base into train, validation and test sets
base_train_tensor, base_val_tensor, base_test_tensor, conditions_train_tensor, conditions_val_tensor, conditions_test_tensor, weights_train_tensor, weights_val_tensor, weights_test_tensor, test_indices_mc = random_splits(
    standardized_base, standardized_conditions, weights, train_fraction=train_fraction, val_fraction=val_fraction
    )

#split the target into train, validation and test sets - independently of the base split (this is not problematic, there is no correspondence between the base and target samples)
target_train_tensor, target_val_tensor, target_test_tensor, test_indices_cms= random_splits(
    standardized_target, train_fraction=train_fraction, val_fraction=val_fraction
)


#----------------------------------------------------------------------initialize torch dataloaders to use in training----------------------------------------------------------------------



'''
#DataLoaders for the weights
weights_train_loader = DataLoader(weights_train_tensor, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
weights_val_loader   = DataLoader(weights_val_tensor,   batch_size=batch_size, num_workers=num_workers,shuffle=False, drop_last=True)
# DataLoaders for the base (MC))
base_train_loader = DataLoader(base_train_tensor, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
base_val_loader   = DataLoader(base_val_tensor,   batch_size=batch_size, num_workers=num_workers,shuffle=False, drop_last=True)
# DataLoaders for conditions
conditions_train_loader = DataLoader(conditions_train_tensor, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
conditions_val_loader   = DataLoader(conditions_val_tensor,   batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
# DataLoaders for the target (CMS)
target_train_loader = DataLoader(target_train_tensor, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
target_val_loader   = DataLoader(target_val_tensor,   batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
'''

#Dataloader for base, conditions and weights
base_train_loader = DataLoader(TensorDataset(base_train_tensor, conditions_train_tensor, weights_train_tensor), batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
base_val_loader = DataLoader(TensorDataset(base_val_tensor, conditions_val_tensor, weights_val_tensor), batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)

#Dataloader for target
target_train_loader = DataLoader(target_train_tensor, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True, generator=generator)
target_val_loader   = DataLoader(target_val_tensor,   batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)


#start the training loop:
start = time.time()
#----------------------------------------------------------------------cfm training loop----------------------------------------------------------------------

if morphing_strategy == 'cfm':
    # Create an instance of the conditional flow matcher
    cfm = ConditionalFlowMatcher(sigma=sigma_cfm)

    for epoch in range(n_epochs):
        print("Epoch: ", epoch+1)

        #arrays storing the loss per batch for the current epoch
        array_loss_train_epoch = []
        array_loss_val_epoch = []



        model.train()
        for step, ((base_batch, conditions_batch, weights_batch) , target_batch) in enumerate(zip(base_train_loader, target_train_loader)):
            optimizer.zero_grad()
            #ensure that the batches are on the correct device
            base_batch = base_batch.to(device)
            conditions_batch = conditions_batch.to(device)
            target_batch = target_batch.to(device)
            weights_batch = weights_batch.to(device)

            # apply the selected OT strategy (ot_strategy string variable from the config) by rearranging the batches
            base_batch, conditions_batch, target_batch, weights_batch = apply_ot_strategy(base_batch, conditions_batch, target_batch, weights_batch, ot_strategy)
    

            '-----------------------------------------------cfm specific--------------------------------------------------'
            #calculate the optimal transport field direction and magnitude ut and its position xt at time t from conditional flow matching
            t, xt, ut = cfm.sample_location_and_conditional_flow(base_batch, target_batch)

            #concatenate the sample location xt, the conditions_batch and t for the model input
            model_input = torch.cat([xt, conditions_batch, t.unsqueeze(-1)], dim=1)
            #note: this fixates the model input to always be a concat of (xt, condition, t)

            #calculate the model output vt
            vt = model(model_input)

            #determine loss as MSE between the computet vector field and the ideal vector field
            loss = loss_function(vt, ut, weights_batch)
            '-------------------------------------------cfm specific completed--------------------------------------------------'

            #call backpropagated computation of gradient in network parameter space
            loss.backward()

            # apply gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            #call step to use the previously calculated gradient to update the model parameters according to the chosen optimizer
            optimizer.step()

            #append loss in the loss array such that loss can be plotted later
            array_loss_train_epoch.append(loss.item())

        array_loss_train.append(np.mean(array_loss_train_epoch))

        

        model.eval()
        #validation loop
        with torch.no_grad():
            for step, ((base_batch, conditions_batch, weights_batch) , target_batch) in enumerate(zip(base_val_loader, target_val_loader)):
                optimizer.zero_grad()
                #ensure that the batches are on the correct device
                base_batch = base_batch.to(device)
                conditions_batch = conditions_batch.to(device)
                target_batch = target_batch.to(device)
                weights_batch = weights_batch.to(device)

                # apply the selected OT strategy
                base_batch, conditions_batch, target_batch, weights_batch = apply_ot_strategy(base_batch, conditions_batch, target_batch, weights_batch, ot_strategy)

                '-----------------------------------------------cfm specific--------------------------------------------------'
                #calculate the optimal transport field direction and magnitude ut and its position xt at time t from conditional flow matching
                t, xt, ut = cfm.sample_location_and_conditional_flow(base_batch, target_batch)

                #concatenate the sample location xt, the conditions_batch and t for the model input
                model_input = torch.cat([xt, conditions_batch, t.unsqueeze(-1)], dim=1)
                #note: this fixates the model input to always be a concat of (xt, condition, t)

                #calculate the model output vt
                vt = model(model_input)

                #determine loss as MSE between the computet vector field and the ideal vector field
                loss = loss_function(vt, ut, weights_batch)
                '-------------------------------------------cfm specific completed--------------------------------------------------'

                #append loss in the loss array such that loss can be plotted later
                array_loss_val_epoch.append(loss.item())

            array_loss_val.append(np.mean(array_loss_val_epoch))
            # Step scheduler based on validation loss
            scheduler.step(array_loss_val[-1])

            # Early stopping for CFM branch
            # (Insert after validation loss computation and scheduler step)
            if epoch == 0:
                best_val_loss = array_loss_val[-1]
                best_val_epoch = 0
                epochs_no_improve = 0
            else:
                current_val_loss = array_loss_val[-1]
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_val_epoch = epoch
                    epochs_no_improve = 0
                    # Save the best model state for CFM branch
                    print(f'new best model found with loss = {best_val_loss}')
                    torch.save(model.state_dict(), os.path.join(out_dir_models, 'best_model.pth'))
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"early stopping triggered at epoch {epoch+1}")
                    break

                print(f"Current validation loss: {current_val_loss}")
                print(f"Best validation loss: {best_val_loss} at epoch {best_val_epoch+1}")
                print(f"Patience status: {early_stop_patience-epochs_no_improve} epochs of no improvement left until early stopping")

            if (epoch+1) % 10 == 0 and save_progress == True:  # Save model every 10 epochs
                torch.save(model.state_dict(), os.path.join(out_dir_models, f"epochs{epoch+1}.pth"))
                lossplot_train(array_loss_train, array_loss_val, epoch+1, best_val_epoch, filename=f"loss_plot_epochs{epoch+1}.png", out_dir=out_dir_models)


            end = time.time()
            print(f"Epoch {epoch+1} completed in {(end-start):0.2f} seconds. ")
            start=end






#----------------------------------------------------------------------regression training loop----------------------------------------------------------------------

elif morphing_strategy == 'regular':
    for epoch in range(n_epochs):
        print("Epoch: ", epoch+1)

        #arrays storing the loss per batch for the current epoch
        array_loss_train_epoch = []
        array_loss_val_epoch = []


        model.train()
        for step, ((base_batch, conditions_batch, weights_batch) , target_batch) in enumerate(zip(base_train_loader, target_train_loader)):
            optimizer.zero_grad()
            #ensure that the batches are on the correct device
            base_batch = base_batch.to(device)
            conditions_batch = conditions_batch.to(device)
            target_batch = target_batch.to(device)
            weights_batch = weights_batch.to(device)

            # apply the selected OT strategy
            base_batch, conditions_batch, target_batch, weights_batch = apply_ot_strategy(base_batch, conditions_batch, target_batch, weights_batch, ot_strategy)
            
            '-----------------------------------------------regression specific--------------------------------------------------'
            #concanatenate the base_batch and conditions_batch to create the input for the model
            base_batch_expansion = torch.cat([base_batch, conditions_batch], dim=1)

            #morph the base_batch using the model
            morphed_base_batch = model(base_batch_expansion)

            #determine loss as MSE between the morphed base batch and the target batch
            loss = loss_function(morphed_base_batch, target_batch, weights_batch)
            '-------------------------------------------regression specific completed--------------------------------------------------'

            #call backpropagated computation of gradient in network parameter space
            loss.backward()

            # apply gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            #call step to use the previously calculated gradient to update the model parameters according to the chosen optimizer
            optimizer.step()

            #append loss in the loss array such that loss can be plotted later
            array_loss_train_epoch.append(loss.item())

        array_loss_train.append(np.mean(array_loss_train_epoch))

        

        model.eval()
        with torch.no_grad():
            for step, ((base_batch, conditions_batch, weights_batch) , target_batch) in enumerate(zip(base_val_loader, target_val_loader)):
                optimizer.zero_grad()
                base_batch = base_batch.to(device)
                conditions_batch = conditions_batch.to(device)
                target_batch = target_batch.to(device)
                weights_batch = weights_batch.to(device)

                # apply the selected OT strategy
                base_batch, conditions_batch, target_batch, weights_batch = apply_ot_strategy(base_batch, conditions_batch, target_batch, weights_batch, ot_strategy)

                '-----------------------------------------------regression specific--------------------------------------------------'
                #concanatenate the base_batch and conditions_batch to create the input for the model
                base_batch_expansion = torch.cat([base_batch, conditions_batch], dim=1)

                #morph the base_batch using the model
                morphed_base_batch = model(base_batch_expansion)

                #determine loss as MSE between the morphed base batch and the target batch
                loss = loss_function(morphed_base_batch, target_batch, weights_batch)
                '-------------------------------------------regression specific completed--------------------------------------------------'

                #append loss in the loss array such that loss can be plotted later
                array_loss_val_epoch.append(loss.item())

        array_loss_val.append(np.mean(array_loss_val_epoch))
        # Step scheduler based on validation loss
        scheduler.step(array_loss_val[-1])
        # Early stopping for regular regression branch
        # (Insert after validation loss computation and scheduler step)
        if epoch == 0:
            best_val_loss = array_loss_val[-1]
            best_val_epoch = 0
            epochs_no_improve = 0
        else:
            current_val_loss = array_loss_val[-1]
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_val_epoch = epoch
                epochs_no_improve = 0
                # Save the best model state for regular branch
                print(f'new best model found with loss = {best_val_loss}')
                torch.save(model.state_dict(), os.path.join(out_dir_models, 'best_model.pth'))
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"early stopping triggered at epoch {epoch+1}")
                break

            print(f"Current validation loss: {current_val_loss}")
            print(f"Best validation loss: {best_val_loss} at epoch {best_val_epoch+1}")
            print(f"Patience status: {early_stop_patience-epochs_no_improve} epochs of no improvement left until early stopping")


        if (epoch+1) % 10 == 0 and save_progress == True:  # Save model every 10 epochs
                torch.save(model.state_dict(), os.path.join(out_dir_models, f"epochs{epoch+1}.pth"))
                lossplot_train(array_loss_train, array_loss_val, epoch+1, best_val_epoch, filename=f"loss_plot_epochs{epoch+1}.png", out_dir=out_dir_models)

        end = time.time()
        print(f"Epoch {epoch+1} completed in {(end-start):0.2f} seconds. ")
        start=end

        


#----------------------------------------------------------------------send results to plot_helper_3.py and plot the corrections----------------------------------------------------------------------
#create the loss plot for the best epoch plots
if morphing_strategy == 'cfm':
    out_dir_plots = f'../plotting/plots_{year}/{morphing_strategy}_{ot_strategy}/{sigma_cfm}_sigma_cfm/{layers}_layers/{hidden_dimensions}_hidden_dimensions/{lr}_learning_rate/{batch_size}_batch_size/best_model'
elif morphing_strategy == 'regular':
    out_dir_plots = f'../plotting/plots_{year}/{morphing_strategy}_{ot_strategy}/{layers}_layers/{hidden_dimensions}_hidden_dimensions/{lr}_learning_rate/{batch_size}_batch_size/best_model'
lossplot_train(array_loss_train, array_loss_val, epoch+1, best_val_epoch+1, filename=f"loss_plot_best_epoch_{best_val_epoch+1}.png", out_dir=out_dir_plots)


#filter the test data events in the dataframes
data_test_df = data_df.iloc[test_indices_cms]
mc_test_df = mc_df.iloc[test_indices_mc]

#build payload to pass to the plot_helper_3.py script
payload = {
    "variables_mc_names":      variables_mc_names,   # MC variable names
    "conditions_names": conditions_names,    # conditions names
    "variables_cms_names":    variables_cms_names,  # CMS variable names
    "data_test_df": data_test_df,  # DataFrame for test data
    "mc_test_df": mc_test_df,      # DataFrame for MC test data
    "eta_regions": eta_regions,  # Placeholder for eta regions
    "data_eta_masks": data_eta_masks,  # Placeholder for data eta masks
    "mc_eta_masks": mc_eta_masks,  # Placeholder for MC eta masks
    "variables_cms_names": variables_cms_names,  # CMS variable names
    "variables_mc_names": variables_mc_names,  # MC variable names
    "conditions_names": conditions_names,  # Conditions names
    "morphing_strategy": morphing_strategy,  # Morphing strategy used
    "ot_strategy": ot_strategy,  # OT strategy used
    "hidden_dimensions": hidden_dimensions,  # Hidden dimensions in the model
    "layers": layers,  # Number of layers in the model
    "lr": lr,  # Learning rate used
    "batch_size": batch_size,  # Batch size used
    "device": device,  # Device used for training
    "event_location": event_location,  # Event location used
    "sigma_cfm": sigma_cfm,  # Sigma for CFM if applicable
    "year": year,  # Year of the data
}
# Launch the plot_helper.py under the plotting environment
helper_script = Path(__file__).parent.parent / "plotting" / "plot_helper_3.py"
cmd = [
    "micromamba", "run", "-n", "test_environment",
    "python", str(helper_script),
]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
pickle.dump(payload, proc.stdin)
proc.stdin.close()
proc.stdin = None
out, err = proc.communicate()
if proc.returncode != 0:
    print("Plot helper failed with error:")
    print(err.decode("utf-8"))
else:
    print("Plot helper completed successfully:")
    print(out.decode("utf-8"))





print("Saved best_model plots to:", out_dir_plots)
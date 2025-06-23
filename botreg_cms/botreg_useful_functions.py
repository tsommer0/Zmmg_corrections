import torch.nn as nn
import torch
import pandas as pd

#define the neural network model for the discontiuous NF morphing
class Regressor2(nn.Module):
    def __init__(self, input_dim=2, output_dim=2):
        super(Regressor2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.model(x)
    

def generate_tensors(df, variables):
    """
    Takes a pandas DataFrame and a list of column names, returns:
      - raw_tensor:          FloatTensor of shape [n_samples, n_features]
      - standardized_tensor: FloatTensor of same shape, zero-mean/unit-std per feature
      - mean:                Tensor of shape [n_features]
      - std:                 Tensor of shape [n_features]
    """
    # 1) Build raw tensor
    raw_tensor = torch.from_numpy(df[variables].to_numpy()).float()

    # 2) Compute feature-wise stats (torch.std default is unbiased; pass unbiased=False if you want population std)
    mean = raw_tensor.mean(dim=0)
    std  = raw_tensor.std(dim=0)

    # 3) Standardize
    standardized_tensor = (raw_tensor - mean) / std

    return raw_tensor, standardized_tensor, mean, std

def unstandardize_tensor(tensor, mean, std):
    '''Takes a tensor and the mean and std of the original data, returns the unstandardized tensor.'''
    return tensor * std + mean

#unbalanced optimal transport function

def uot_matching(base_batch, conditions_batch, target_batch, reg=0.01, reg_m=0, tolerance=0.1):
    '''
    Takes batches of base and target distributions and applies unbalanced optimal transport hard matching to give a rearranged base- and target-batch suited for training.
    '''
    
    M = torch.cdist(base_batch, target_batch) ** 2  # Shape: (batch_size, batch_size)
    global M_cpu_buf
    # copy squared distances into pre-allocated NumPy buffer
    M_cpu_t = M.detach().cpu()
    np.copyto(M_cpu_buf, M_cpu_t.numpy())

    # generate histograms to feed into the POT unbalanced sinkhorn
    a = np.ones(M_cpu_buf.shape[0]) / M_cpu_buf.shape[0]

    # generate OT Plan using the pre-allocated buffer
    P = pot.unbalanced.sinkhorn_unbalanced(a, a, M_cpu_buf, reg=reg, reg_m=reg_m)
    
    row_sums = P.sum(axis=1)
    

    # compute hard assignment: for each source, pick target with highest coupling mass
    indices = np.argmax(P, axis=1)

    # convert to torch tensor on base_batch's device
    indices = torch.from_numpy(indices).long().to(base_batch.device)
    #determine which coupling is too small
    unmatch = row_sums < a*tolerance
    #delete indices of couplings below the threshold
    indices = indices[~unmatch]
    # reorder target batch to match to the base batch
    target_batch = target_batch[indices]
    #filter out the unmatched base batch entries
    base_batch = base_batch[~unmatch]
    conditions_batch = conditions_batch[~unmatch]
    

    #return the accepted couplings for training
    return base_batch, conditions_batch, target_batch
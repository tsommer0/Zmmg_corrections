#%%
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
from torch import Tensor

#%%
def circle(n_samples, r=1, sig=.025):
    phi_gen = np.random.uniform(0, 2*np.pi, size=n_samples)
    r_gen = np.random.normal(loc=r, scale=sig, size=n_samples)
    
    x = np.cos(phi_gen)*r_gen
    y = np.sin(phi_gen)*r_gen
    
    out = np.column_stack((x, y))
    
    return Tensor(out)

def four_circles_spread(n_samples, r=1, sig=.025, spread=.5):
    x = circle(n_samples, r, sig)

    n_circ = n_samples/4

    shift = spread*np.array([[1,1], [1,-1], [-1,1], [-1,-1]])

    for i in range(4):
        i_low = int(i*n_circ)
        i_high = int((i+1)*n_circ)
        x[i_low:i_high] = x[i_low:i_high] + shift[i]
        
    return x

def checker(n_samples, n_rows=4, n_columns=4, range_x=[-1,1], range_y=[-1,1], uneven_indices=1):
    
    start_x = range_x[0]
    start_y = range_y[0]
    
    if not (uneven_indices==0 or uneven_indices==1):
        raise ValueError("uneven_indices can only be 1 (represents True) or 0 (represents False)")
    
    range_x = (range_x[1]-range_x[0])/n_columns
    shift_x = np.random.uniform(0, range_x, size=n_samples)
    #column = np.random.randint(0, n_columns-1, size=n_samples)
    
    
    range_y = (range_y[1]-range_y[0])/n_rows
    shift_y = np.random.uniform(0, range_y, size=n_samples)
    #row = np.random.randint(0, n_rows-1, size=n_samples)
    
    cells = np.array([
    (i, j)
    for i in range(n_columns)
    for j in range(n_rows)
    if (i + j) % 2 == uneven_indices
    ])
    
    indices = np.random.choice(len(cells), size=n_samples, replace=True)
    cells = cells[indices]
    
    columns = cells[:,0]
    rows = cells[:,1]
    
    x = range_x*rows + shift_x + start_x
    y = range_y*columns + shift_y + start_y
    
    out = np.column_stack((x, y))
    
    return Tensor(out)

# %%
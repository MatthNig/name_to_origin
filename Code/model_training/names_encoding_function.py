###################################
# function for one-hot-encoding a #
# sequence of names at the level  #
# of characters. The result is a  #
# 3D-Tensor of encoded names      #
###################################

import numpy as np
import pandas as pd

def encode_chars(names, seq_max, char_dict, n_chars):

    N = len(names)
    END_idx = np.where(pd.Series(char_dict) == "END")[0][0]
    
    # Create 3D-Tensor with shape (No. of samples, maximum name length, number of characters):
    tmp = np.zeros(shape = (N, seq_max, n_chars)) 

    # iterate over all names
    for i in range(N):
        name = names[i]
        
        # truncate at seq_max
        if(len(name) > seq_max):
            name = name[:seq_max]
        
        # encode characters
        for char in range(len(name)):
            idx_pos = np.where(pd.Series(char_dict) == name[char])[0][0]
            tmp[i, char, idx_pos] = 1
            
        # padding
        if len(name) < seq_max:
            tmp[i, len(name):seq_max, END_idx] = 1
    
    return tmp
import os
import re
import numpy as np

if __name__=='__main__':
    backprop = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_rnn_backprop_2_layer_and_forward\784\0"
    backprop_files = os.listdir(backprop)
    for i in backprop_files:
        directory = os.path.join(backprop, i)
        file = directory + '/params.txt'
        vals = []
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                accuracy_pos = line.find('Accuracy:')
                accuracy_val = line[accuracy_pos+10:accuracy_pos + 15]
                vals.append(float(accuracy_val))
        vals_np = np.array(vals)
        np.savetxt(directory + '/acc_meta.txt', vals_np, fmt='%f')
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    results_dir = os.getcwd() + "/results"
    save_dir = os.getcwd() + "/graphs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    mode_1_bias = results_dir + "/Mode_1/mode_1_bias"
    individual_bias_dir = results_dir + "/individual_maybe/1"
    individual_bias_results = os.listdir(individual_bias_dir)
    
    individual_bias_results = [individual_bias_dir + "/" + x for x in individual_bias_results][6:] + [mode_1_bias]
    label = ["Chemical_{} Individual".format(i) for i in range(1, 6)] + ["5 Chemicals Non Individual"]

    three_chem = individual_bias_results[2] + "/UpdateWeights.pth"
    print(three_chem)
    three_chem_bias = torch.load(three_chem, map_location=torch.device('cpu'), weights_only=True)["bias_dictionary.chemical2"]

    print(three_chem_bias)
    three_chem_bias = np.array(three_chem_bias, dtype=np.float32, copy=True)
    print(three_chem_bias.shape)
    #plt.matshow(three_chem_bias[0], cmap='seismic')#
    #plt.colorbar()
    #plt.matshow(three_chem_bias[1], cmap='seismic')

    sign = np.sign(three_chem_bias[0])
    sign_2 = np.sign(three_chem_bias[1])
    plt.matshow(sign-sign_2, cmap='seismic')
    #plt.matshow(three_chem_bias[2], cmap='seismic')

    plt.matshow(three_chem_bias[0] - three_chem_bias[1], cmap='seismic')
    plt.colorbar()
    plt.show()

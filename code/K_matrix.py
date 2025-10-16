import os

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch

if __name__ == "__main__":
    results_dir = os.getcwd() + "/results"
    save_dir = os.getcwd() + "/rnns_graphs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    """five_chem = (
        results_dir + r"/DFA_5_regularized_not_K/0/20251009-180024/UpdateWeights.pth"
    )"""

    five_chem = results_dir + r"/DFA_longer_5/0/20251008-023058/UpdateWeights.pth"
    five_chem_K_matrix = torch.load(
        five_chem, map_location=torch.device("cpu"), weights_only=True
    )["K_matrix"]

    five_chem_K_matrix = np.array(five_chem_K_matrix, dtype=np.float32, copy=True)
    print(five_chem_K_matrix)
    print(np.max(np.abs(five_chem_K_matrix)))
    divnorm = colors.TwoSlopeNorm(vmin=-0.02, vcenter=0.0, vmax=0.02)

    plt.figure(figsize=(4, 3.5))
    plt.matshow(five_chem_K_matrix, cmap="seismic", norm=divnorm, fignum=1)
    plt.colorbar(norm=divnorm, cmap="seismic")
    plt.xlabel("Complex Synapse State")
    plt.ylabel("Complex Synapse State")
    plt.yticks(
        ticks=np.arange(0, 5),
        labels=np.arange(1, 6),
    )
    plt.title(
        "K Matrix for 5 State Direct Vector-Error Feedback", fontdict={"fontsize": 10}
    )
    plt.savefig(save_dir + "/K_matrix_5_chem.png")
    plt.show()

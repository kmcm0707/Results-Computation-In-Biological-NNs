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

    five_chem = (
        results_dir + r"/DFA_5_regularized_not_K/0/20251009-180024/UpdateWeights.pth"
    )
    five_chem_v_vector = torch.load(
        five_chem, map_location=torch.device("cpu"), weights_only=True
    )["v_vector"]

    five_chem_v_vector = torch.softmax(five_chem_v_vector, dim=1)

    print(five_chem_v_vector)

    five_chem_v_vector = np.array(five_chem_v_vector, dtype=np.float32, copy=True)

    divnorm = colors.TwoSlopeNorm(vmin=-0.012, vcenter=0.0, vmax=0.3)

    plt.figure(figsize=(4, 3))
    plt.matshow(five_chem_v_vector, cmap="seismic", fignum=1, norm=divnorm)
    plt.xticks(np.arange(-0.5, 5, 1), minor=True)
    plt.grid(which="minor", color="black", linestyle="-", linewidth=1)
    plt.title(
        "V vector for 5 State Complex Synapse",
        fontdict={"fontsize": 10},
    )
    plt.yticks([], [])

    plt.savefig(save_dir + "/v_vector_5_chem.png")
    plt.show()

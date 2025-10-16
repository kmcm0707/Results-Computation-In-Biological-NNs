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
    five_chem_P_matrix = torch.load(
        five_chem, map_location=torch.device("cpu"), weights_only=True
    )["P_matrix"]

    five_chem_P_matrix = np.array(five_chem_P_matrix, dtype=np.float32, copy=True)
    # remove columns 5
    five_chem_P_matrix = np.delete(five_chem_P_matrix, np.s_[5], axis=1)
    # remove column 7
    five_chem_P_matrix = np.delete(five_chem_P_matrix, np.s_[7], axis=1)
    # remove column 7
    five_chem_P_matrix = np.delete(five_chem_P_matrix, np.s_[6], axis=1)
    divnorm = colors.TwoSlopeNorm(vmin=-0.012, vcenter=0.0, vmax=0.012)

    plt.figure(figsize=(4, 3))
    plt.matshow(five_chem_P_matrix, cmap="seismic", norm=divnorm, fignum=1)
    plt.colorbar(norm=divnorm, cmap="seismic")
    plt.title(
        "P Matrix for 5 State Direct Vector-Error Feedback",
        fontdict={"fontsize": 10},
    )
    plt.xlabel("Plasticity Motif")
    plt.ylabel("Complex Synapse State")
    plt.xticks(
        ticks=np.arange(0, 7),
        labels=["$m_1$", "$m_2$", "$m_3$", "$m_4$", "$m_5$", "$m_6$", "$m_7$"],
    )

    plt.savefig(save_dir + "/P_matrix_5_chem.png")
    plt.show()

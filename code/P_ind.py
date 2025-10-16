import os

import matplotlib.pyplot as plt
import numpy as np
import torch

if __name__ == "__main__":
    runner_mode_6_ind_non_ind_v = r"results\DFA_5_regularized_not_K\0\20251009-180024\acc_meta.txt"
    state_dict = torch.load(
        runner_mode_6_ind_non_ind_v + r"\UpdateWeights.pth",
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    P_lay_1 = state_dict["P_dictionary.chemical1"]
    P_lay_2 = state_dict["P_dictionary.chemical2"]
    P_lay_3 = state_dict["P_dictionary.chemical3"]
    P_lay_4 = state_dict["P_dictionary.chemical4"]
    P_lay_5 = state_dict["P_dictionary.chemical5"]

    P = [
        P_lay_1,
        P_lay_2,
        P_lay_3,
        P_lay_4,
        P_lay_5,
    ]  # Assuming these are the matrices for each chemical
    colors = [
        "red",
        "blue",
        "fuchsia",
        "lime",
        "cyan",
        "purple",
        "orange",
        "black",
        "yellow",
        "green",
    ]

    plt.rc("font", family="serif", size=11)
    plt.figure(figsize=(10, 6))
    all_layers_i_chem_1 = []
    for i in P:
        i_chem_1 = i[0, :].numpy()
        i_chem_1 = [i_chem_1[0:5], i_chem_1[-2:]]
        i_chem_1_flat = [item for sublist in i_chem_1 for item in sublist]
        all_layers_i_chem_1.append(i_chem_1_flat)
    all_layers_i_chem_1 = np.array(all_layers_i_chem_1)
    print(all_layers_i_chem_1.shape)
    layers_index = [1, 2, 3, 4, 5]
    for i in range(7):
        plt.plot(
            layers_index,
            all_layers_i_chem_1[:, i],
            marker="o",
            color=colors[i],
            label=f"Plasticity Motif {i + 1}",
        )

    plt.legend()
    plt.xlabel("Layer Index")
    plt.xticks(layers_index)
    plt.ylabel("P Matrix Weightings for Chemical 1")
    plt.title(r"P Matrix Weightings for Chemical 1 in LD-WN-RCN")
    plt.tight_layout()

    save_dir = os.getcwd() + "/graphs/"
    plt.savefig(save_dir + "ld_wn_p_1.png")
    plt.show()

    all_layers_i_chem_2 = []
    for i in P:
        i_chem_2 = i[1, :].numpy()
        i_chem_2 = [i_chem_2[0:5], i_chem_2[-2:]]
        i_chem_2_flat = [item for sublist in i_chem_2 for item in sublist]
        all_layers_i_chem_2.append(i_chem_2_flat)
    all_layers_i_chem_2 = np.array(all_layers_i_chem_2)
    print(all_layers_i_chem_2.shape)
    layers_index = [1, 2, 3, 4, 5]
    for i in range(7):
        plt.plot(
            layers_index,
            all_layers_i_chem_2[:, i],
            marker="o",
            color=colors[i],
            label=f"Plasticity Motif {i + 1}",
        )
    plt.legend()
    plt.xlabel("Layer Index")
    plt.xticks(layers_index)
    plt.ylabel("P Matrix Weightings for Chemical 2")
    plt.title(r"P Matrix Weightings for Chemical 2 in LD-WN-RCN")
    plt.tight_layout()
    save_dir = os.getcwd() + "/graphs/"
    plt.savefig(save_dir + "ld_wn_p_2.png")
    plt.show()

    all_layers_i_chem_3 = []
    for i in P:
        i_chem_3 = i[2, :].numpy()
        print(i_chem_3)
        i_chem_3 = [i_chem_3[0:5], i_chem_3[-2:]]
        print(i_chem_3)
        i_chem_3_flat = [item for sublist in i_chem_3 for item in sublist]
        all_layers_i_chem_3.append(i_chem_3_flat)

    all_layers_i_chem_3 = np.array(all_layers_i_chem_3)
    print(all_layers_i_chem_3.shape)
    layers_index = [1, 2, 3, 4, 5]
    for i in range(7):
        plt.plot(
            layers_index,
            all_layers_i_chem_3[:, i],
            marker="o",
            color=colors[i],
            label=f"Plasticity Motif {i + 1}",
        )
    plt.legend()
    plt.xlabel("Layer Index")
    plt.xticks(layers_index)
    plt.ylabel("P Matrix Weightings for Chemical 3")
    plt.title(r"P Matrix Weightings for Chemical 3 in LD-WN-RCN")
    plt.tight_layout()

    save_dir = os.getcwd() + "/graphs/"
    plt.savefig(save_dir + "ld_wn_p_3.png")
    plt.show()

    # plt.ylim(-1e-5, 1e-5)

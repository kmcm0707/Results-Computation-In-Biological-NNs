import os

import matplotlib.pyplot as plt
import numpy as np
import torch

if __name__ == "__main__":
    # runner_director = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results\mode_4\different_y_0\min_tau_testing"
    # end = [r"\1_1_50", r"\2", r"\3", r"\4", r"\5", r"\20", r"\30", r"\50"]
    # all_folders = []
    # for i in end:
    #    all_folders.append(runner_director + i)

    # labels = ["Backprop", "Shervani-Tabar", "RCN"]
    # labels = ["1.5", "2", "3", "4", "5", "20", "30", "50"]
    # int_labels = [1.5, 2, 3, 4, 5, 20, 30, 50]
    mode_6_3_chem = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_weight_mode_6\0"  # 3 chem
    mode_6_7_chem_tau_500 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_7_chem_mode_6_800_min_tau_500\0"
    mode_6_tau_1000 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results\mode_6\normalise_mode_6_fix\0\1000"
    mode_6_tau_500 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results\mode_6\normalise_mode_6_fix\0\500"

    mode_6_5_chem_100_tau = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results\mode_6\normalise_mode_6_5_chem_2\0\100"
    mode_6_5_chem_500_tau = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results\mode_6\normalise_mode_6_5_chem_2\0\500"
    mode_6_5_chem_1000_tau = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results\mode_6\normalise_mode_6_5_chem_2\0\1000"

    labels = ["100", "500", "1000"]
    all_folders = [mode_6_5_chem_100_tau, mode_6_5_chem_500_tau, mode_6_5_chem_1000_tau]
    int_labels = [100, 500, 1000]

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

    plt.rc("font", family="serif", size=13)
    # smallest_eigenvalues = []
    large_eigenvalues_real = []
    large_eigenvalues_imag = []
    large_eigenvalues_combined = []
    fig, ax = plt.subplots(figsize=(10, 6))
    # ax.set_ylim(-1e-19, 1e-19)
    # axins = ax.inset_axes([0.1, 0.6, 0.8, 0.3])  # [x, y, width, height]
    ##axins.set_xlim(-1e-2, 1e-2)
    # axins.set_ylim(-1e-20, 1e-20)

    for index, i in enumerate(all_folders):
        state_dict = torch.load(
            i + r"\UpdateWeights.pth",
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        K_matrix = state_dict["K_matrix"]
        K_matrix = K_matrix.cpu().numpy()
        # smallest_eigenvalue = np.min(np.real(np.linalg.eigvals(K_matrix)))
        large_eigenvalue = np.max(np.real(np.linalg.eigvals(K_matrix)))
        large_eigenvalue_imag = np.max(np.imag(np.linalg.eigvals(K_matrix)))
        large_eigenvalues_real.append(large_eigenvalue + int_labels[index])
        large_eigenvalues_imag.append(large_eigenvalue_imag)

        min_tau = 2
        max_tau = int_labels[index]
        base = max_tau / min_tau

        tau_vector = min_tau * (base ** torch.linspace(0, 1, 5))
        z_vector = 1 / tau_vector
        y_vector = 1 - z_vector

        diag_y = np.diag(y_vector)

        K_y = diag_y + K_matrix - np.eye(K_matrix.shape[0])
        all_eigenvalues = np.linalg.eigvals(K_y)
        print(np.max(np.real(all_eigenvalues)))
        ax.scatter(
            np.real(all_eigenvalues),
            np.imag(all_eigenvalues),
            color=colors[index],
            label=labels[index],
        )
        """axins.scatter(
            np.real(all_eigenvalues),
            np.imag(all_eigenvalues),
            color=colors[index],
        )"""

    # ax.indicate_inset_zoom(axins, edgecolor="black")
    plt.legend()
    plt.xlabel("Real Eigenvalue")
    plt.ylabel("Imaginary Eigenvalue")
    plt.title(r"Eigenvalues of $-Y+K$ Matrix for Different $\tau_{min}$ Values")
    plt.tight_layout()

    save_dir = os.getcwd() + "/graphs/"
    plt.savefig(save_dir + "eigenvalues_mode_6.png")
    plt.show()
    """plt.plot(
        int_labels,
        large_eigenvalues_real,
        marker="o",
        color="red",
        label="Real Eigenvalue",
    )"""
    """plt.plot(
        int_labels,
        large_eigenvalues_combined,
        marker="o",
        color="blue",
        label="Combined Eigenvalue",
    )"""

    # plt.ylim(-1e-5, 1e-5)

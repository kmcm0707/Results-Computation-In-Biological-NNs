import os

import matplotlib.pyplot as plt
import numpy as np
import torch

if __name__ == "__main__":
    results_dir = os.getcwd() + "/results_runner"
    name = "mode_9_chem_analysis"
    path = results_dir + r"/runner_mode_9_chem_analysis/0/250"
    chemical_autocorrelation = path + "/chemical_autocorrelation"
    all_files = os.listdir(path)
    save_dir = os.getcwd() + "/rnns_graphs/"
    if not os.path.exists(save_dir + f"/{name}/"):
        os.makedirs(save_dir + f"/{name}/")
    save_dir = save_dir + f"/{name}/"

    """
    all_autocorr = []
    for file in all_files:
        if "chemical_autocorrelation" in file:
            with open(path + "/" + file, "r") as f:
                all_lines = f.readlines()
                # index of lines containing "chemical1"
                chem_1_indices = [
                    i for i, line in enumerate(all_lines) if "chemical1" in line
                ]
                all_data = []
                for i in range(len(chem_1_indices) - 1):
                    start = chem_1_indices[i] + 2
                    end = chem_1_indices[i + 1]
                    data = np.loadtxt(all_lines[start:end:2])
                    all_data.append(data)
                all_autocorr.append(np.mean(all_data, axis=0))
    parameter_autocorr = []
    for file in all_files:
        if "parameter_autocorrelation" in file and "chemical" not in file:
            with open(path + "/" + file, "r") as f:
                all_lines = f.readlines()
                # index of lines containing "chemical1"
                parameter_indices = [
                    i for i, line in enumerate(all_lines) if "Parameter" in line
                ]
                all_data = []
                for i in range(len(parameter_indices) - 1):
                    start = parameter_indices[i] + 2
                    end = parameter_indices[i + 1]
                    data = np.loadtxt(all_lines[start:end:2])
                    all_data.append(data)
                parameter_autocorr.append(np.mean(all_data, axis=0))

    plt.rc("font", family="serif", size=18)
    fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(18, 5), sharey=True)
    labels = [
        r"SV 1",
        r"SV 2",
        r"SV 3",
        r"SV 4",
        r"SV 5",
        r"Parameter",
    ]
    colors = [
        "#0343df",
        "#e50000",
        "#f97306",
        "#7e1e9c",
        "#008000",
        "#18FFB2",
        "#ffa500",
        "#000000",
        "#ffff00",
        "#008000",
    ]

    x_axis = np.arange(0, int(all_autocorr[0].shape[0]))

    for i in range(5):
        for j in range(5):
            ax[i].plot(
                x_axis[0:800],
                all_autocorr[i][:, j][0:800],
                label=labels[j],
                color=colors[j],
            )
            # ax[i].scatter(x_axis[0:800], all_autocorr[i][:, j][0:800], color=colors[j])
    for i in range(5):
        ax[i].plot(
            x_axis[0:800],
            parameter_autocorr[i][0:800],
            label=labels[5],
            color=colors[5],
        )
        # ax[i].scatter(x_axis[0:800], parameter_autocorr[i][0:800], color=colors[5])
    ax[0].set_title("Layer 1")
    ax[0].set_xlabel("Training Samples")
    ax[0].set_ylabel("Autocorrelation (degrees)")

    # ax[0].set_ylim(0, 99)
    # ax[0].set_yticks(np.arange(0, 100, 10))

    ax[0].legend()

    ax[1].set_title("Layer 2")
    ax[1].set_xlabel("Training Samples")
    # ax[1].set_ylabel("Degrees")
    # ax[1].set_ylim(0, 99)
    # ax[0,1].legend()

    ax[2].set_title("Layer 3")
    ax[2].set_xlabel("Training Samples")
    # ax[2].set_ylabel("Degrees")
    # ax[2].set_ylim(0, 99)
    # ax[1,0].legend()

    ax[3].set_title("Layer 4")
    ax[3].set_xlabel("Training Samples")

    ax[4].set_title("Layer 5")
    ax[4].set_xlabel("Training Samples")
    # ax[3].set_ylabel("Degrees")
    # ax[3].set_ylim(0, 99)
    # ax[1,1].legend()

    # fig.suptitle(
    #    "5 State Complex Synapse"
    # )  # Angle between error signals and backprop for Meta-trained models
    # fig.tight_layout()
    

    plt.tight_layout()
    plt.savefig(save_dir + f"{name}_autocorrelation_angle.png")
    plt.show()"""

    """all_actual_autocorr = {}
    temp_data = []
    lags = [1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50]
    index = 0
    for file in all_files:
        if "chemical_actual_autocorrelation" in file:
            current_lag = file.split("_")[-3]
            index += 1
            with open(path + "/" + file, "r") as f:
                all_lines = f.readlines()
                # index of lines containing "chemical1"
                chem_1_indices = [
                    i for i, line in enumerate(all_lines) if "chemical1" in line
                ]
                all_data = []
                for i in range(len(chem_1_indices) - 1):
                    start = chem_1_indices[i] + 2
                    end = chem_1_indices[i + 1]
                    data = np.loadtxt(all_lines[start:end:2])
                    all_data.append(data)
                temp_data.append(np.mean(all_data, axis=0))
            if index == 5:
                all_actual_autocorr[current_lag] = temp_data
                index = 0
                temp_data = []"""

    """if not os.path.exists(save_dir + f"/{name}/"):
        os.makedirs(save_dir + f"/{name}/")
    for lag in lags:
        fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(18, 5), sharey=True)
        for i in range(5):
            for j in range(5):
                ax[i].plot(
                    x_axis[0:800],
                    all_actual_autocorr[str(lag)][i][:, j][0:800],
                    label=labels[j],
                    color=colors[j],
                )
            ax[i].set_title(f"Layer {i + 1} - Lag {lag}")
            ax[i].set_xlabel("Training Samples")
            if i == 0:
                ax[i].set_ylabel("Autocorrelation")
                ax[i].legend()
        plt.tight_layout()
        plt.savefig(save_dir + f"actual_autocorr_lag_{lag}.png")
        # plt.show()"""

    """pre_cuttoff = 250
    mean_autocorr = {}
    for lag in lags:
        mean_autocorr[lag] = []
        for layer in range(5):
            mean_value = np.mean(
                all_actual_autocorr[str(lag)][layer][:pre_cuttoff, :], axis=0
            )
            mean_autocorr[lag].append(mean_value)
        mean_autocorr[lag] = np.array(mean_autocorr[lag])

    layer1_means = []
    for lag in lags:
        layer1_means.append(mean_autocorr[lag][0, :])
    layer1_means = np.array(layer1_means)
    layer2_means = []
    for lag in lags:
        layer2_means.append(mean_autocorr[lag][1, :])
    layer2_means = np.array(layer2_means)
    layer3_means = []
    for lag in lags:
        layer3_means.append(mean_autocorr[lag][2, :])
    layer3_means = np.array(layer3_means)
    layer4_means = []
    for lag in lags:
        layer4_means.append(mean_autocorr[lag][3, :])
    layer4_means = np.array(layer4_means)
    layer5_means = []
    for lag in lags:
        layer5_means.append(mean_autocorr[lag][4, :])
    layer5_means = np.array(layer5_means)
    layer_means = [
        layer1_means,
        layer2_means,
        layer3_means,
        layer4_means,
        layer5_means,
    ]

    fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(18, 5), sharey=True)
    for i in range(5):
        for j in range(5):
            ax[i].plot(
                lags,
                layer_means[i][:, j],
                label=labels[j],
                color=colors[j],
            )

        ax[i].set_xlabel("Lag")
        if i == 0:
            ax[i].set_ylabel("Mean Autocorrelation")
            ax[i].legend()
    plt.tight_layout()
    plt.savefig(save_dir + f"{name}_mean_autocorrelation.png")"""

    """chemical_tracking = []
    for file in all_files:
        if "chemical_tracking" in file:
            with open(path + "/" + file, "r") as f:
                # print(f"Processing {file}")
                all_lines = f.readlines()
                split_indices = [
                    # i for i, line in enumerate(all_lines) if "Parameter_Numbers" in line
                    0,
                    1250,
                    2250,
                    3250,
                    4250,
                    5250,
                    6250,
                ]
                all_data = []
                for i in range(len(split_indices) - 1):
                    start = split_indices[i] + 1
                    end = split_indices[i + 1]
                    data = np.loadtxt(all_lines[start:end])
                    all_data.append(data[0:800, :])
                    break  # Only need first segment
                chemical_tracking.append(np.mean(all_data, axis=0))

    parameter_tracking = []
    for file in all_files:
        if "parameter_tracking" in file:
            with open(path + "/" + file, "r") as f:
                # print(f"Processing {file}")
                all_lines = f.readlines()
                split_indices = [
                    0,
                    1250,
                    2250,
                    # i for i, line in enumerate(all_lines) if "Parameter_Numbers" in line
                ]
                all_data = []
                for i in range(len(split_indices) - 1):
                    start = split_indices[i] + 1
                    end = split_indices[i + 1]
                    data = np.loadtxt(all_lines[start:end])
                    all_data.append(data[0:800, :])
                    break  # Only need first segment
                parameter_tracking.append(np.mean(all_data, axis=0))

    x_axis = np.arange(0, int(chemical_tracking[0].shape[0]))
    print(chemical_tracking[0].shape)
    fig, ax = plt.subplots(figsize=(18, 10), ncols=6, nrows=5, sharex=True, sharey=True)

    for i in range(5):
        for j in range(5):
            ax[i, j].plot(
                x_axis,
                chemical_tracking[i * 5 + j][:, 0:6],
            )
    for i in range(5):
        ax[i, 5].plot(
            x_axis,
            parameter_tracking[i][:, 0:6],
        )

    ax[0, 0].set_ylabel("Layer 1")
    ax[1, 0].set_ylabel("Layer 2")
    ax[2, 0].set_ylabel("Layer 3")
    ax[3, 0].set_ylabel("Layer 4")
    ax[4, 0].set_ylabel("Layer 5")
    for j in range(5):
        ax[4, j].set_xlabel(f"SV {j + 1}")
    ax[4, 5].set_xlabel("Parameter")
    plt.tight_layout()
    plt.savefig(save_dir + f"{name}_chemical_parameter_tracking.png")
    plt.show()"""

    Kh_tracking = []
    for file in all_files:
        if "Kh_tracking" in file:
            with open(path + "/" + file, "r") as f:
                print(f"Processing {file}")
                all_lines = f.readlines()
                split_indices = [
                    # i for i, line in enumerate(all_lines) if "Parameter_Numbers" in line
                    0,
                    1250,
                    2250,
                ]
                all_data = []
                for i in range(len(split_indices) - 1):
                    start = split_indices[i] + 1
                    end = split_indices[i + 1]
                    data = np.loadtxt(all_lines[start:end])
                    all_data.append(data[0:800, :])
                    break  # Only need first segment
                Kh_tracking.append(np.mean(all_data, axis=0))

    x_axis = np.arange(0, int(Kh_tracking[0].shape[0]))
    fig, ax = plt.subplots(ncols=5, nrows=5, figsize=(18, 10), sharey=True, sharex=True)
    for i in range(5):
        for j in range(5):
            ax[i, j].plot(
                x_axis,
                Kh_tracking[i * 5 + j][:, :],
            )
    ax[0, 0].set_ylabel("Layer 1")
    ax[1, 0].set_ylabel("Layer 2")
    ax[2, 0].set_ylabel("Layer 3")
    ax[3, 0].set_ylabel("Layer 4")
    ax[4, 0].set_ylabel("Layer 5")
    ax[4, 0].set_xlabel("SV 1")
    ax[4, 1].set_xlabel("SV 2")
    ax[4, 2].set_xlabel("SV 3")
    ax[4, 3].set_xlabel("SV 4")
    ax[4, 4].set_xlabel("SV 5")
    plt.tight_layout()
    plt.savefig(save_dir + "Kh_tracking.png")
    plt.show()

    Pf_tracking = []
    for file in all_files:
        if "Pf_tracking" in file:
            with open(path + "/" + file, "r") as f:
                print(f"Processing {file}")
                all_lines = f.readlines()
                split_indices = [
                    # i for i, line in enumerate(all_lines) if "Parameter_Numbers" in line
                    0,
                    1250,
                    2250,
                    3250,
                ]
                all_data = []
                for i in range(len(split_indices) - 1):
                    start = split_indices[i] + 1
                    end = split_indices[i + 1]
                    data = np.loadtxt(all_lines[start:end])
                    all_data.append(data[0:800, :])
                    break  # Only need first segment
                Pf_tracking.append(np.mean(all_data, axis=0))

    x_axis = np.arange(0, int(Pf_tracking[0].shape[0]))
    fig, ax = plt.subplots(ncols=5, nrows=5, figsize=(18, 13), sharex=True)
    for i in range(5):
        for j in range(5):
            ax[i, j].plot(
                x_axis,
                Pf_tracking[i * 5 + j][:, :],
            )
    ax[0, 0].set_ylabel("Layer 1")
    ax[1, 0].set_ylabel("Layer 2")
    ax[2, 0].set_ylabel("Layer 3")
    ax[3, 0].set_ylabel("Layer 4")
    ax[4, 0].set_ylabel("Layer 5")
    ax[4, 0].set_xlabel("SV 1")
    ax[4, 1].set_xlabel("SV 2")
    ax[4, 2].set_xlabel("SV 3")
    ax[4, 3].set_xlabel("SV 4")
    ax[4, 4].set_xlabel("SV 5")
    plt.tight_layout()
    plt.savefig(save_dir + f"{name}_Pf_tracking.png")
    plt.show()

    chemical_norms = []
    for file in all_files:
        if "chemical_norms" in file:
            with open(path + "/" + file, "r") as f:
                print(f"Processing {file}")
                all_lines = f.readlines()
                split_indices = [
                    i for i, line in enumerate(all_lines) if "Timestep: 1 " in line
                ]
                all_data = []
                for i in range(len(split_indices) - 1):
                    start = split_indices[i] + 1
                    end = split_indices[i + 1]
                    data = np.loadtxt(all_lines[start:end:2])
                    all_data.append(data)
                chemical_norms.append(np.mean(all_data, axis=0))
    # parameter_norms = []
    """for file in all_files:
        if "parameter_norms" in file:
            with open(path + "/" + file, "r") as f:
                print(f"Processing {file}")
                all_lines = f.readlines()
                split_indices = [
                    i for i, line in enumerate(all_lines) if "Timestep: 1" in line
                ]
                all_data = []
                for i in range(len(split_indices) - 1):
                    start = split_indices[i] + 1
                    end = split_indices[i + 1]
                    data = np.loadtxt(all_lines[start:end:2])
                    all_data.append(data)
                parameter_norms.append(np.mean(all_data, axis=0))"""

    limit = 800
    x_axis = np.arange(0, int(chemical_norms[0].shape[0]))
    fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(18, 5), sharey=True)
    for i in range(5):
        for j in range(5):
            ax[i].plot(
                x_axis[0:limit],
                chemical_norms[i][:, j][0:limit],
                label=f"SV {j + 1}",
            )
        ax[i].set_title(f"Layer {i + 1}")
        ax[i].set_xlabel("Training Samples")
        ax[i].set_ylabel("Norms")
        if i == 0:
            ax[i].legend()
        """ax[i].plot(
            x_axis[0:limit],
            parameter_norms[i][0:limit],
            label="Parameter Norms",
            color="black",
            linestyle="--",
        )"""
    plt.tight_layout()
    plt.savefig(save_dir + f"{name}_chemical_norms.png")
    plt.show()

    model = torch.load(
        path + "/UpdateWeights.pth", map_location=torch.device("cpu"), weights_only=True
    )
    v_vector = model["v_vector"]
    print(v_vector)
    softmax_v = torch.nn.Softmax(dim=1)(v_vector)
    print(softmax_v)

    x_axis = np.arange(0, int(chemical_norms[0].shape[0]))
    fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(18, 5), sharey=True)
    for i in range(5):
        for j in range(5):
            ax[i].plot(
                x_axis[0:limit],
                chemical_norms[i][:, j][0:limit] * softmax_v[0][j].item(),
                label=f"SV {j + 1}",
            )
        ax[i].set_title(f"Layer {i + 1}")
        ax[i].set_xlabel("Training Samples")
        ax[i].set_ylabel(r"$\pi \cdot$Norms")
        if i == 0:
            ax[i].legend()
        """ax[i].plot(
            x_axis[0:limit],
            parameter_norms[i][0:limit],
            label="Parameter Norms",
            color="black",
            linestyle="--",
        )"""
    plt.tight_layout()
    plt.savefig(save_dir + f"{name}_chemical_norms_pi.png")
    plt.show()

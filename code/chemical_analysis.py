import os

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    results_dir = os.getcwd() + "/results_runner"
    path = results_dir + r"/mode_7_chem_analysis_50/0/250"
    chemical_autocorrelation = path + "/chemical_autocorrelation"

    all_files = os.listdir(path)
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
        if "parameter_autocorrelation" in file:
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
        r"Chemical 1",
        r"Chemical 2",
        r"Chemical 3",
        r"Chemical 4",
        r"Chemical 5",
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
    save_dir = os.getcwd() + "/rnns_graphs/"
    plt.tight_layout()
    plt.savefig(save_dir + "autocorrelation_angle.png")
    plt.show()

    chemical_tracking = []
    for file in all_files:
        if "chemical_tracking" in file:
            with open(path + "/" + file, "r") as f:
                # print(f"Processing {file}")
                all_lines = f.readlines()
                split_indices = [
                    i for i, line in enumerate(all_lines) if "Parameter_Numbers" in line
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
                    i for i, line in enumerate(all_lines) if "Parameter_Numbers" in line
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
    fig, ax = plt.subplots(figsize=(10, 6), ncols=6, nrows=5, sharex=True, sharey=True)

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
        ax[4, j].set_xlabel(f"Chemical {j + 1}")
    ax[4, 5].set_xlabel("Parameter")
    plt.tight_layout()
    plt.savefig(save_dir + "chemical_parameter_tracking.png")
    plt.show()

    Kh_tracking = []
    for file in all_files:
        if "Kh_tracking" in file:
            with open(path + "/" + file, "r") as f:
                print(f"Processing {file}")
                all_lines = f.readlines()
                split_indices = [
                    i for i, line in enumerate(all_lines) if "Parameter_Numbers" in line
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
                Kh_tracking[i * 5 + j][:, 0:6],
            )
    ax[0, 0].set_ylabel("Layer 1")
    ax[1, 0].set_ylabel("Layer 2")
    ax[2, 0].set_ylabel("Layer 3")
    ax[3, 0].set_ylabel("Layer 4")
    ax[4, 0].set_ylabel("Layer 5")
    ax[4, 0].set_xlabel("Chemical 1")
    ax[4, 1].set_xlabel("Chemical 2")
    ax[4, 2].set_xlabel("Chemical 3")
    ax[4, 3].set_xlabel("Chemical 4")
    ax[4, 4].set_xlabel("Chemical 5")
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
                    i for i, line in enumerate(all_lines) if "Parameter_Numbers" in line
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
                Pf_tracking[i * 5 + j][:, 0:6],
            )
    ax[0, 0].set_ylabel("Layer 1")
    ax[1, 0].set_ylabel("Layer 2")
    ax[2, 0].set_ylabel("Layer 3")
    ax[3, 0].set_ylabel("Layer 4")
    ax[4, 0].set_ylabel("Layer 5")
    ax[4, 0].set_xlabel("Chemical 1")
    ax[4, 1].set_xlabel("Chemical 2")
    ax[4, 2].set_xlabel("Chemical 3")
    ax[4, 3].set_xlabel("Chemical 4")
    ax[4, 4].set_xlabel("Chemical 5")
    plt.tight_layout()
    plt.savefig(save_dir + "Pf_tracking.png")
    plt.show()

import os

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    save_dir = os.getcwd() + "/rnns_graphs/"
    results_dir = os.getcwd() + "/results_runner"
    backprop = results_dir + "/runner_backprop_CL_2/0/"
    runner_backprop_CL_EWC_3 = results_dir + "/runner_backprop_CL_EWC_3/0/"
    runner_backprop_CL_EWC_2000 = results_dir + "/runner_backprop_CL_EWC_2000/0/"
    runner_backprop_CL_EWC_10000 = results_dir + "/runner_backprop_CL_EWC_10000/0/"
    mode_9_5_chem = results_dir + "/runner_mode_9_CB/0/"
    mode_9_1_chem = results_dir + "/runner_mode_9_CB_1_chem/0/"

    limit = 50
    runner_folders = [
        backprop,
        # runner_backprop_CL_EWC_3,
        # runner_backprop_CL_EWC_2000,
        runner_backprop_CL_EWC_10000,
        mode_9_1_chem,
        mode_9_5_chem,
    ]
    labels = [
        "BP",
        "EWC",
        # "EWC 10000",
        # "EWC 2000",
        "1 SV",
        "5 SV",
    ]
    colors = [
        "#0343df",
        "#e50000",
        "#7e1e9c",
        "#f97306",
        "#008000",
        "#800080",
        "#ffa500",
        "#000000",
        "#ffff00",
        "#008000",
    ]
    plt.rc("font", family="serif", size=14)
    plt.figure(figsize=(12, 8))
    for index, i in enumerate(runner_folders):
        all_files = os.listdir(i)
        all_files_int = [int(all_files[i]) for i in range(len(all_files))]
        all_files_int.sort()
        for k in range(len(all_files_int)):
            if all_files_int[k] > limit:
                all_files_int = all_files_int[:k]
                break

        try:
            all_files_int.remove(1)
            all_files_int.remove(2)
            all_files_int.remove(3)
            all_files_int.remove(4)
            # all_files_int.remove(5)
            all_files_int.remove(6)
            all_files_int.remove(7)
            all_files_int.remove(8)
            all_files_int.remove(9)
        except ValueError:
            pass

        all_files = [str(x) for x in all_files_int]
        all_values_1 = np.array([])
        all_values_2 = np.array([])
        for j in all_files:
            directory = i + "/" + j + "/params.txt"
            z_1 = []
            z_2 = []
            with open(directory, "r") as f:
                lines = f.readlines()

                if "backprop" in i:
                    accuracy = "Accuracy:"
                    current = "Current"
                    accuracy_2 = "Accuracy 2:"
                else:
                    accuracy = "Accuracy:"
                    current = "Current"
                    accuracy_2 = "Accuracy_2:"

                for line in lines:
                    if accuracy in line:
                        if "backprop" in i:
                            accuracy_index = line.find(accuracy)
                            current_index = line.find(current)
                            accuracy_2_index = line.find(accuracy_2)
                            current_index_2 = len(line)
                        else:
                            accuracy_index = line.find(accuracy)
                            current_index = line.find(current)
                            accuracy_2_index = line.find(accuracy_2)
                            current_index_2 = line.find(current, accuracy_2_index)

                        acc_1 = line[
                            accuracy_index + len(accuracy) : current_index
                        ].strip()
                        acc_2 = line[
                            accuracy_2_index + len(accuracy_2) : current_index_2
                        ].strip()
                        z_1.append(float(acc_1))
                        z_2.append(float(acc_2))

            average_1 = np.mean(z_1, axis=0)
            std_1 = np.std(z_1, axis=0)
            # median = np.median(z, axis=0)
            all_values_1 = np.append(all_values_1, average_1)
            average_2 = np.mean(z_2, axis=0)
            std_2 = np.std(z_2, axis=0)
            # median = np.median(z, axis=0)
            all_values_2 = np.append(all_values_2, average_2)
            # print(i, j, average, std)
        all_values_1 = all_values_1[:limit]
        all_values_2 = all_values_2[:limit]
        x_axis = all_files_int[: len(all_values_1)]

        plt.plot(
            x_axis,
            all_values_1,
            label=labels[index],  # + " (EMNIST)",
            color=colors[index],
        )
        plt.plot(
            x_axis,
            all_values_2,
            linestyle="--",
            color=colors[index],
            # label=labels[index] + " (F-MNIST)",
        )
        plt.scatter(x_axis, all_values_1, color=colors[index])
        plt.scatter(x_axis, all_values_2, color=colors[index])

    plt.vlines(70.5, 0, limit, colors="black", label="EMNIST")
    plt.vlines(70.5, 0, limit, colors="black", linestyles="--", label="F-MNIST")
    plt.xlabel("Fashion MNIST Samples Per Class")
    plt.ylabel("Accuracy (%)")
    plt.ylim(-0.02, 1)
    plt.xlim(-2, limit + 5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir + "continued_learning_emnist_fashionmnist.png", dpi=300)
    plt.show()

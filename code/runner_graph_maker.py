import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional

if __name__ == '__main__':
    
    runner_diff_y0_3_chems = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_diff_y0_3_chems\0"
    runner_individual_no_bias = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_individual_no_bias\0"
    runner_different_y_ind_v_diff_lr = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_different_y_ind_v_diff_lr\0"
    runner_super_varied = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_super_varied\0"

    runner_folders = [runner_diff_y0_3_chems, runner_individual_no_bias, runner_different_y_ind_v_diff_lr, runner_super_varied]
    labels = [r"$\tau_{min}=2$", r"Layer Dependent $\tau_{min}=1$", r"Layer Dependent $\tau_{min}=2$", r"$\tau_{min}=2$ Meta Learner Changed"]
    colors = ["blue", "fuchsia", "lime", "red"]

    plt.rc('font', family='serif', size=14)
    plt.figure(figsize=(13, 8))
    for index, i in enumerate(runner_folders):
        all_files = os.listdir(i)
        all_files_int = [int(all_files[i] ) for i in range(len(all_files))]
        all_files_int.sort()
        all_files = [str(x) for x in all_files_int]
        all_values = np.array([])
        for j in all_files:
            directory = i + "/" + j
            z = np.loadtxt(directory + "/acc_meta.txt")
            average = np.mean(z, axis=0)
            all_values = np.append(all_values, average)
        x_axis = np.arange(1, len(all_values)+1) * 10
        plt.plot(x_axis, all_values, label=labels[index], color=colors[index])
        #plt.fill_between(x_axis, 0, all_values, alpha=0.5, facecolor=colors[index])
        plt.scatter(x_axis, all_values, color=colors[index])
    plt.legend()
    plt.title("Average Accuracy Per Training Images Per Class for 3 Chemical Trained RCNs")
    plt.xlabel("Training Images Per Class")
    plt.xlim(0, 200)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    save_dir = os.getcwd() + "/graphs/"
    plt.savefig(save_dir + "runner_graph.png")
    plt.show()


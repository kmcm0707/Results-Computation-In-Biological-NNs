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
    runner_y0_3_extra_long_500 = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_y0_3_extra_long_500\0"
    runner_y0_3_extra_long800 = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_y0_3_extra_long_800\0"
    runner_rosenbaum_50 = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_rosenbaum_50\0"
    runner_rosenbaum_varied = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_rosenbaum_varied\0"
    runner_y0_4_extra_long_100 = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_y0_4_extra_long_100\0"
    runner_y0_4_extra_long_120 = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_y0_4_extra_long_120\1"
    runner_lr_5 = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_lr_5\0"

    runner_director = "C:\\Users\\Kyle\\Desktop\\Results-Computation-In-Biological-NNs/results_runner/runner_min_tau_testing_"
    end = ["1_1_50", "2", "3", "4", "5", "10", "20", "30", "40", "50"]
    all_folders = []
    for i in end:
        all_folders.append(runner_director + i + "/0")

    #print(all_folders)
    


    runner_folders = [runner_super_varied, runner_y0_3_extra_long800, runner_lr_5, runner_rosenbaum_varied]
    labels = [r"$\tau_{min}=50$", r"$\tau_{min}=100$", r"$\tau_{min}=100$ New Learning Rule", "Rosenbaum"]
    colors = ["red", "blue", "fuchsia", "lime", "cyan", "purple", "orange", "black", "yellow", "green"]

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
            #median = np.median(z, axis=0)
            all_values = np.append(all_values, average)
        all_values = all_values
        x_axis = all_files_int
        x_axis = x_axis
        plt.plot(x_axis, all_values, label=labels[index], color=colors[index])
        #plt.fill_between(x_axis, 0, all_values, alpha=0.5, facecolor=colors[index])
        plt.scatter(x_axis, all_values, color=colors[index])
    plt.legend()
    plt.title("Average Accuracy Per Training Images Per Class for Meta-Trained RCNs")
    plt.xlabel("Training Images Per Class")
    plt.xlim(0, 400)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    save_dir = os.getcwd() + "/graphs/"
    plt.savefig(save_dir + "runner_lr_5_00.png")
    plt.show()


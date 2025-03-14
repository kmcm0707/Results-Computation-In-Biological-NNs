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
    runner_super_varied = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_super_varied\0"
    runner_y0_3_extra_long_500 = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_y0_3_extra_long_500\0"
    runner_y0_3_extra_long800 = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_y0_3_extra_long_800\0"
    runner_rosenbaum_50 = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_rosenbaum_50\0"
    runner_rosenbaum_varied = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\rosenbaum\runner_rosenbaum_varied\0"
    runner_y0_4_extra_long_100 = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_y0_4_extra_long_100\0"
    runner_y0_4_extra_long_120 = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_y0_4_extra_long_120\1"
    runner_lr_5 = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_lr_5\0"

    runner_4_chems = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_y0_4_extra_long_200\0"
    runner_5_chems_500_gpu = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_y0_5_extra_long_200\runner_500_epochs_gpu\0"
    runner_5_chems_500_cpu = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_y0_5_extra_long_200\runner_500_epochs_cpu\0"
    runner_post_train = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_y0_5_extra_long_200\runner_200_epochs_post_train_30_70\0"
    runner_100_post_train = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_y0_5_extra_long_200\runner_100_epochs_post_train_30_70\0"

    """runner_director = "C:\\Users\\Kyle\\Desktop\\Results-Computation-In-Biological-NNs/results_runner/runner_min_tau_testing_"
    end = ["1_1_50", "2", "3", "4", "5", "10", "20", "30", "40", "50"]
    all_folders = []
    for i in end:
        all_folders.append(runner_director + i + "/0")"""
    
    runner_DFA_grad_500= r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_DFA_grad_test\0"
    runner_DFA_grad= r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_DFA_grad_800\0"
    runner_DFA = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_DFA_test\0"
    runnner_FA_no_Grad = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_FA_No_Grad_Test\0"

    runner_eta = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_eta\0"

    runner_combined = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_Combined\0"
    runner_combined_2 = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_Combined_v3\0"

    runner_rosenbaum_fashion_mnist = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_rosenbaum_fashion_mnist\0"
    runner_3chem_fashion_mnist = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_3chem_fashion_mnist\0"
    runner_3chem_fashion_mnist2 = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_3chem_fashion_mnist2\0"

    backprop = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_backprop\0"

    normalised_dodge_5 = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\mode_5\runner_normalised_weight_3chem_new5a\0"
    normalised_non_dodge_5 = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\mode_5\runner_normalised_3_chem\0"
    normalised_5_chem = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\mode_5\runner_normalised_weight_5_chem_800\0"

    fixed_normalised = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_normalise_weight_mode_6\0"
    fixed_normalised_feedback = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_mode6_wtih_feedback\0"
    fixed_normalised_5_chem = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_5_chem_mode_6\0"
    fixed_normalised_5_chem_800 = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_5_chem_mode_6_800_2\0"
    

    #print(all_folders)
    runner_folders = [runner_rosenbaum_varied, backprop, fixed_normalised, fixed_normalised_5_chem_800, runner_DFA_grad]
    labels = [r"Rosenbaum", r"Backprop", r"Normalised FA RCN and Normalised Weights", r"Normalised FA RCN and Normalised Weights 5 Chemicals", r"DFA"]
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
    plt.vlines(30, 0.1, 0.9, colors="black", linestyles="dashed", label="Training region for Normalised RCN")
    plt.vlines(110, 0.1, 0.9, colors="black", linestyles="dashed")
    plt.vlines(30, 0.1, 0.9, colors="blue", linestyles="dashed", label="Training region for Normal RCN")
    plt.vlines(150, 0.1, 0.9, colors="blue", linestyles="dashed")
    plt.legend()
    plt.title("Average Accuracy Per Training Images Per Class")
    plt.xlabel("Training Images Per Class")
    plt.xlim(0, 400)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    save_dir = os.getcwd() + "/graphs/"
    plt.savefig(save_dir + "mode_6_5_chem_DFA_400.png")
    plt.show()


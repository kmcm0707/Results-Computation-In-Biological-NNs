import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional


def multi_plot_accuracy(directories, labels, window_size=20, save_dir=None, save_name="accuracy"):
    """
        Plot the meta accuracy using a moving average.

    The method first computes a moving average of the meta accuracy values
    stored in a text file located in the results directory. It then plots
    the moving average values against the meta-training episodes. Finally,
    the plot is saved to a file in the results directory.

    :return: None
    """
    # -- plot
    plt.figure()
    average = []
    for directory in directories:
        z = np.loadtxt(directory + "/acc_meta.txt")
        z = comp_moving_avg(np.nan_to_num(z), window_size)
        z = z[0:400]
        average = average + [z]
    smallest = min([len(x) for x in average])
    average = [x[:smallest] for x in average]
    average = np.array(average)
    x = np.array(range(average.shape[1])) + int((window_size - 1) / 2)
    print(x.shape, average.shape)
    for i in range(len(directories)):
        plt.plot(x, average[i], label="{} last={:.2f}".format(labels[i], average[i][-1]))
    plt.axhline(y=0.2, color="r", linestyle="--", label="Chance Level")
    plt.xlabel("Meta-Training Episodes")
    plt.ylabel("Meta Accuracy")
    plt.title("Meta Accuracy (Average)")
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(save_dir + "/" + save_name, bbox_inches="tight")
    plt.close()

def multi_plot_loss(directories, labels, window_size=20, save_dir=None, save_name="loss"):
    """
        Plot the meta loss using a moving average.

    The method first computes a moving average of the meta loss values
    stored in a text file located in the results directory. It then plots
    the moving average values against the meta-training episodes. Finally,
    the plot is saved to a file in the results directory.

    :return: None
    """
    # -- plot
    plt.figure()
    average = []
    for directory in directories:
        z = np.loadtxt(directory + "/loss_meta.txt")
        z = z[0:200]
        z = comp_moving_avg(np.nan_to_num(z), window_size)

        average = average + [z]
    average = np.array(average)
    x = np.array(range(average.shape[1])) + int((window_size - 1) / 2)
    print(x.shape, average.shape)
    for i in range(len(directories)):
        plt.plot(x, average[i], label="{} last={:.2f}".format(labels[i], average[i][-1]))

    plt.ylim([0, 10])
    plt.xlabel("Meta-Training Episodes")
    plt.ylabel("Meta Loss")
    plt.title("Meta Loss (Average)")
    plt.legend()
    plt.savefig(save_dir + "/" + save_name, bbox_inches="tight")
    plt.close()

def peak_meta_accuracy_scatter_plot(directories, labels, save_dir=None, save_name="peak_meta_accuracy", window_size=20):
    plt.rc('font', family='serif', size=10)
    M = len(directories)
    xx = np.linspace(0, M, M)
    plt.figure(figsize=(10, 5))
    #sorted_indices = np.argsort(mean_player_skills)
    #sorted_names = W[sorted_indices]

    average = []
    for directory in directories:
        z = np.loadtxt(directory + "/acc_meta.txt")
        z = z[0:200]
        z = comp_moving_avg(np.nan_to_num(z), window_size)

        average = average + [z]
    average = np.array(average)
    best_x = np.max(average, axis=1)

    plt.scatter(xx, best_x, color='red', s=200, marker='x')
    plt.ylim([np.min(best_x) - 0.05, np.max(best_x) + 0.05])
    #plt.errorbar(mean_skill[sorted_indices], xx, xerr=std_dev[sorted_indices], fmt='o', label='Gibbs Sampling std', color='red', capsize=10, elinewidth=5, capthick=8)
    #plt.errorbar(mean_player_skills[sorted_indices], xx, xerr=1/np.sqrt(precision_player_skills[sorted_indices]), fmt='o', label='Message Passing std', color='blue', alpha = 0.5, capsize=10, elinewidth=5, capthick=8)
    #plt.scatter(mean_player_skills[sorted_indices], xx, label='Message Passing', s=200, marker='o')
    plt.xticks(np.linspace(0, M, M), labels=labels)
    plt.xlabel('Model')
    plt.ylabel('Peak Meta-Training Episode Accuracy (Average)')
    plt.title('Peak Meta-Training Episode Accuracy (Average over {} episodes)'.format(window_size))
    plt.grid()
    plt.savefig(save_dir + "/" + save_name, bbox_inches="tight")

def matrix_plot(matrix, title, save_dir=None):
    """
        Plot a matrix.

    The method plots a matrix and saves the plot to a file in the results
    directory.

    :return: None
    """
#plt.imshow(K, cmap='seismic', vmin=-np.max(np.abs(K)), vmax=+np.max(np.abs(K)))
 
    plt.figure()
    plt.matshow(matrix, cmap="seismic")
    plt.title(title)
    plt.colorbar()
    plt.savefig(save_dir + "/matrix_plot", bbox_inches="tight")
    plt.close()


def comp_moving_avg(data, window_size):
    """
        Compute the moving average of a dataset.

    The method computes the moving average of a dataset using a window
    size. The window size determines the number of data points that are
    averaged to produce a single value. The method returns the moving
    average values.

    :return: The moving average values.
    """
    return np.convolve(data, np.ones(window_size), "valid") / window_size


if __name__ == "__main__":
    results_dir = os.getcwd() + "/results"
    save_dir = os.getcwd() + "/graphs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    mode_1_bias = results_dir + "/Mode_1/mode_1_bias"
    individual_bias_dir = results_dir + "/individual_maybe/1"
    individual_bias_results = os.listdir(individual_bias_dir)
    
    individual_bias_results = [individual_bias_dir + "/" + x for x in individual_bias_results][6:] + [mode_1_bias]
    label = ["Chemical_{} Individual".format(i) for i in range(1, 6)] + ["5 Chemicals Non Individual"]

    mode_1 = results_dir + "/Mode_1/add"
    mode_1_500 = results_dir + "/Mode_1/add_500"
    different_non_lins_dir = results_dir + "/different_non_lins/0"
    different_non_lins_results = os.listdir(different_non_lins_dir)
    different_non_lins_results = [different_non_lins_dir + "/" + x for x in different_non_lins_results] + [mode_1]
    label = ["ReLU", "ELU", "Leaky ReLU", "GELU", "Tanh"]

    benna_fusi_dir_0 = results_dir + "/benna/0"
    benna_fusi_dir_1 = results_dir + "/benna/1"
    benna_fusi_results_0 = os.listdir(benna_fusi_dir_0)
    benna_fusi_results_1 = os.listdir(benna_fusi_dir_1)
    benna_fusi_results_0 = [benna_fusi_dir_0 + "/" + x for x in benna_fusi_results_0][:-1]
    benna_fusi_results_1 = [benna_fusi_dir_1 + "/" + x for x in benna_fusi_results_1]
    benna_fusi_results = benna_fusi_results_0 + benna_fusi_results_1
    label = ["Benna Fusi 0", "Benna Fusi 1"]

    individual_no_bias = results_dir + "/individual_no_bias/1"
    individual_no_bias_results = os.listdir(individual_no_bias)
    individual_no_bias_results = [individual_no_bias + "/" + x for x in individual_no_bias_results][0:3] + [mode_1]
    del individual_no_bias_results[1]
    label = ["Chemical 3 Individual,\nlr: 0.0005", "Chemical 3 Individual, lr: 0.001"] + ["5 Chemicals Non Individual"]

    attention = results_dir + "/attention/0"
    attention_results = os.listdir(attention)
    attention_results = [attention + "/" + x for x in attention_results] + [mode_1_500]
    label = ['$v=(Ah+b)(Bh+b)$ {} Chemicals'.format(i) for i in range(2, 8)] + ["5 Chemicals Non Attention"]

    attention_2 = results_dir + "/attention_2/0"
    attention_results_2 = os.listdir(attention_2)
    attention_results_2 = [attention_2 + "/" + x for x in attention_results_2] + [mode_1_500]
    label = ['{} Chemicals'.format(i) for i in range(2, 8)] + ["5 Chemicals Non Attention"]

    attention_3 = results_dir + "/attention_3/0/20250108-181522"
    attention_3_results = [attention_3, mode_1_500]
    label = ['$v=(A(h, update)+b)(B(h, update)+b)$ 3 Chemicals'] + ["5 Chemicals Non Attention"]

    mode_2 = results_dir + "/Mode_2/"
    mode_2_results = os.listdir(mode_2)
    mode_2_results = [mode_2 + x for x in mode_2_results][0:2]
    individual_vs_non = individual_bias_results[2:4] + mode_2_results
    #label = ["2 Chemicals Indivdual", "3 Chemicals Indivdual", "2 Chemicals Non Individual", "3 Chemicals Non Individual"]
    
    multi_plot_accuracy(attention_3_results, label, window_size=20, save_dir=save_dir, save_name="attention_3.png")
    peak_meta_accuracy_scatter_plot(individual_vs_non, label, save_dir=save_dir, save_name="individual_vs_non.png")
    #multi_plot_loss(benna_fusi_results, label, window_size=20, save_dir=save_dir, save_name="benna_fusi_loss.png")
    print("Done")


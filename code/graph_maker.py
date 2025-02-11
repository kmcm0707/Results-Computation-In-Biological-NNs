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
        z = z
        average = average + [z]
    smallest = min([len(x) for x in average]) 
    average = [x[:smallest] for x in average]
    average = np.array(average)
    x = np.array(range(average.shape[1])) + int((window_size - 1) / 2)
    print(x.shape, average.shape, len(labels))
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
    plt.rc('font', family='serif', size=14)
    plt.figure(figsize=(10, 5))
    average = []
    for directory in directories:
        z = np.loadtxt(directory + "/loss_meta.txt")
        z = z[0:450]
        z = comp_moving_avg(np.nan_to_num(z), window_size)

        average = average + [z]
    average = np.array(average)
    x = np.array(range(average.shape[1])) + int((window_size - 1) / 2)
    print(x.shape, average.shape)
    for i in range(len(directories)):
        plt.plot(x, average[i], label="{} last={:.2f}".format(labels[i], average[i][-1]))

    plt.ylim([0, 4])
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
        z = z[0:450]
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

def meta_accuracy_scatter_plot(directories, labels, save_dir=None, save_name="meta_accuracy", inital_cutoff=0, window_size=20, final_cutoff=200):
    plt.rc('font', family='serif', size=14)
    M = len(directories)
    xx = np.linspace(0, M, M)
    plt.figure(figsize=(20, 5))
    #sorted_indices = np.argsort(mean_player_skills)
    #sorted_names = W[sorted_indices]

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'black', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'yellow']
    average = []
    index = 0
    for directory in directories:
        z = np.loadtxt(directory + "/acc_meta.txt")
        if index == 2:
            z = z[619:669]
        else:
            z = z[inital_cutoff:final_cutoff]
        z = comp_moving_avg(np.nan_to_num(z), window_size)
        average = average + [z]
        index += 1

    average = np.array(average)
    average_x = np.mean(average, axis=1)
    std_dev = np.std(average, axis=1)

    print(average_x, std_dev)

    for i in range(len(directories)):
        plt.scatter(xx[i], average_x[i], s=200, marker='x', label=labels[i] + " mean={:.2f}".format(average_x[i]), c=colors[i])
        plt.errorbar(xx[i], average_x[i], yerr=std_dev[i], capsize=2, elinewidth=2, capthick=2, fmt='x', c=colors[i])
    #plt.errorbar(xx, average_x, yerr=std_dev, color='red', capsize=2, elinewidth=2, capthick=2, fmt='x')
    plt.xticks(np.linspace(0, M, M), labels=labels)
    plt.xlabel('Model')
    plt.ylabel('Meta-Training Episode Accuracy (Average)')
    plt.title('Meta-Training Episode Accuracy (Average over {} episodes)'.format(window_size))
    plt.legend()
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
    layer_1 = [[ 0.00303983 , -0.00397467, 0.01357314],
        [-0.00619128,  0.01326045, 0.0015361 ]] 
    layer_2 = [[-0.00855512, -0.00674491, 0.0153236 ],
            [-0.01498219, -0.01596152,  0.00651816]] 
    layer_3 = [[ 0.00667669, -0.00257127, 0.01131714],
            [ 0.01044864, -0.00993618, 0.00202428]]
    layer_4 = [[-0.00274365,  -0.00679861, 0.00440654],
        [-0.0118851, -0.00399994,  0.00671344]]
    layer_5 = [[ 0.01694941, 0.00064761, -0.00233966],
        [-0.07566587, -0.02978626, -0.03679708]]
    
    """layer_1 = [[ 0.04355053, -0.00799247, 0.00189623],
 [ 0.02642857, -0.01639876, -0.00840973],
 [ 0.04979671, -0.01617483,  -0.00339308]]
    layer_2 = [[ 0.01962904,   -0.02863633,  0.00492339],
 [ 0.0090639, -0.03150699, -0.00580211],
 [ 0.02135146, -0.03126799,  0.00070722]] 
    layer_3 = [[ 0.00695393,  -0.0295288,  0.00476656],
 [ 0.00210635, -0.033152 ,  0.00170687],
 [ 0.01071094, -0.03275613,  0.00378403]] 
    layer_4 = [[-0.00038449, -0.03307956,  0.00681321],
            [-0.00593754, -0.03832917, -0.00257698],
            [-0.00141296, -0.03782933,  0.00264486]] 
    layer_5 = [[ 0.01389605,  0.00786305, 0.00130177],
 [-0.012261, -0.01006908, 0.00818825],
 [-0.00796888, -0.01034107, -0.00295752]] """
    layers = [layer_1, layer_2, layer_3, layer_4, layer_5]
    max_val = np.max([np.max(np.abs(i)) for i in layers])
    layer_1 = [[-0.00301073,  0.01338324],
 [ 0.01620886,  0.05991842]] 
    layer_2 = [[-0.00135528,  0.00779566],
 [ 0.02161961,  0.05046608]] 
    layer_3 = [[-0.00061672,  0.00403009],
 [ 0.02029086,  0.04342334]] 
    layer_4 = [[0.00293646, 0.00248897],
 [0.02841949, 0.03633319]]
    layer_5 = [[ 0.0066308,   0.02309774],
 [-0.02356016,  0.01703037]] 
    layers = [layer_1, layer_2, layer_3, layer_4, layer_5]
    #max_val = np.max([np.max(np.abs(i)) for i in layers])
    index = 0
    plt.figure(figsize=(25, 10))
    #plt.title(title)
    plt.rc('font', family='serif', size=14)
    fig, axs = plt.subplots(1, 5)
    for i in layers:
        
        axs[index].imshow(i, cmap='seismic', vmin=-max_val, vmax=max_val)
        if index == 2:
            axs[index].set_title("Layer Dependent RCN $K$ Matrix" + "\n Layer {}".format(index))
        else:
            axs[index].set_title("Layer {}".format(index + 1))
        
        if index == 2:
            #axs[index].set_xticks(range(3), ['$F^0$', '$F^1$ \n Learning Rules', '$F^2$'])
            axs[index].set_xticks(range(2), ['1 \n \t Chemical', '2'])
        else:
            #axs[index].set_xticks(range(3), ['$F^0$', '$F^1$', '$F^2$'])
            axs[index].set_xticks(range(2), ['1', '2'])
        if index == 0:
            #axs[index].set_yticks(range(2), ['Chemical 1', 'Chemical 2'])
            axs[index].set_yticks(range(2), ['Chemical 1', 'Chemical 2'])
        else:
            axs[index].set_yticks([], [])
        index += 1
    
    norm = plt.Normalize(vmin=-max_val, vmax=max_val)
    sm = plt.cm.ScalarMappable(cmap='seismic', norm=norm)
    cax = cax = fig.add_axes([axs[4].get_position().x1 + 0.01,axs[4].get_position().y0,0.02,axs[4].get_position().y1-axs[4].get_position().y0])
    fig.colorbar(sm, ax=axs, cax=cax)
    #plt.suptitle(title)
    #plt.tight_layout()
    plt.savefig(save_dir + "/" + title, bbox_inches="tight")



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

def meta_accuracy_scatter_plot_2(directories, labels, save_dir=None, save_name="meta_accuracy", inital_cutoff=0, window_size=20, final_cutoff=200, color='red', legend=""):
    M = len(directories)
    xx = np.linspace(0, M, M)
    #sorted_indices = np.argsort(mean_player_skills)
    #sorted_names = W[sorted_indices]

    average = []
    for directory in directories:
        z = np.loadtxt(directory + "/acc_meta.txt")
        z = z[inital_cutoff:final_cutoff]
        z = comp_moving_avg(np.nan_to_num(z), window_size)

        average = average + [z]
    average = np.array(average)
    average_x = np.mean(average, axis=1)
    std_dev = np.std(average, axis=1)

    plt.scatter(xx[0], average_x[0], color="purple", s=200, marker='x')
    plt.scatter(xx[1:], average_x[1:], color=color, s=200, marker='x', label=legend)
    plt.ylim([np.min(average_x - std_dev) - 0.05, np.max(average_x + std_dev) + 0.05])
    print(average_x, std_dev)
    plt.errorbar(xx[0], average_x[0], yerr=std_dev[0], color="purple", capsize=2, elinewidth=2, capthick=2, fmt='x')
    plt.errorbar(xx[1:], average_x[1:], yerr=std_dev[1:], color=color, capsize=2, elinewidth=2, capthick=2, fmt='x')
    plt.xticks(np.linspace(0, M, M), labels=labels)

if __name__ == "__main__":
    results_dir = os.getcwd() + "/results"
    save_dir = os.getcwd() + "/graphs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    old_rosenbaum = results_dir + "/Baselines/rosenbaum_old/All_Enabled/2024-10-07_17-50-20_20"

    """mode_1_bias = results_dir + "/Mode_1/mode_1_bias"
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

    old_rosenbaum = results_dir + "/Baselines/rosenbaum_old/All_Enabled/2024-10-07_17-50-20_20"

    individual_no_bias = results_dir + "/individual_no_bias/1"
    individual_no_bias_results = os.listdir(individual_no_bias)
    individual_no_bias_results = [individual_no_bias + "/" + x for x in individual_no_bias_results][0:3] + [mode_1, old_rosenbaum]
    del individual_no_bias_results[1]
    
    label = ["Chemical 3 Individual,\nlr: 0.0005", "Chemical 3 Individual, lr: 0.001"] + ["5 Chemicals Non Individual"] + ["Shervani Tabar"]

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
    label = ["2 Chemicals Indivdual", "3 Chemicals Indivdual", "2 Chemicals Non Individual", "3 Chemicals Non Individual", "Shervani Tabar"]

    old_rosenbaum = results_dir + "/Baselines/rosenbaum_old/All_Enabled/2024-10-07_17-50-20_20"
    individual_vs_non = individual_vs_non + [old_rosenbaum]


    individual_no_bias = results_dir + "/individual_no_bias/1"
    individual_no_bias_results = os.listdir(individual_no_bias)
    result_3_indvidual_no_bias = [individual_no_bias + "/" + x for x in individual_no_bias_results][0]
    label = ["Shervani-Tabar"] + ["{} Chemicals".format(i) for i in range(2, 6)]

    more_individual = results_dir + "/more_individual/0"
    more_individual_results = os.listdir(more_individual)[-3:]
    more_individual_results = [old_rosenbaum] + [more_individual + "/" + x for x in more_individual_results]
    more_individual_results.insert(2, result_3_indvidual_no_bias)
    
    mode_2_bias = results_dir + "/Mode_2/"
    mode_2_bias_results = os.listdir(mode_2_bias)
    mode_2_bias_results = [mode_2_bias + x for x in mode_2_bias_results]
    mode_2_bias_results = mode_2_bias_results[0:2] + [mode_2_bias_results[-2]]
    mode_2_bias_results = [old_rosenbaum] + mode_2_bias_results
    label = ["Shervani Tabar", "2 Chemicals", "3 Chemicals", "5 Chemicals"]

    mode_3 = "results/Mode_3/20241126-225253"
    mode_3_results = [old_rosenbaum, mode_3]
    label = ["Shervani Tabar", "momentum-RCN"]

    more_mode_3 = "results/mode_3_h_zero/0"
    more_mode_3_results = os.listdir(more_mode_3)
    mode_3_results = [old_rosenbaum] + [more_mode_3 + "/" + x for x in more_mode_3_results]
    label = ["Shervani Tabar"] + ["h_zero chemicals={}".format(i) for i in range(1, 6)]
    print(len(mode_3_results))
    print(len(label))

    mode_3_h_same = "results/mode_3_h_same/0"
    mode_3_results = os.listdir(mode_3_h_same)
    mode_3_results = [mode_3_h_same + "/" + x for x in mode_3_results]
    mode_3_results = [old_rosenbaum] + mode_3_results
    label = ["Shervani Tabar"] + ["h_same chemicals={}".format(i) for i in range(1, 6)]

    mode_3_v_change = "results/mode_3_v_change/0"
    mode_3_results = os.listdir(mode_3_v_change)
    mode_3_results = [mode_3_v_change + "/" + x for x in mode_3_results]
    mode_3_results = [old_rosenbaum] + mode_3_results
    label = ["Shervani Tabar"] + ["v_change chemicals={}".format(i) for i in range(1, 6)]

    mode_3_ind = "results/mode_3_ind/0"
    mode_3_results = os.listdir(mode_3_ind)
    mode_3_results = [mode_3_ind + "/" + x for x in mode_3_results][2:5] + [[mode_3_ind + "/" + x for x in mode_3_results][-1]]
    mode_3_results = mode_3_results
    label = ["ind chemicals={}".format(i) for i in range(2, 6)]

    different_y_0 = "results/different_y_0/0"
    different_y_0_results = os.listdir(different_y_0)
    different_y_0_results = [different_y_0 + "/" + x for x in different_y_0_results][-2:]
    mode_3_results = [old_rosenbaum] + different_y_0_results
    label = ["Shervani Tabar", "different_y 3 chems", "different_y 5 chems"]"""
    

    beta = "results/mode_3_all/mode3_beta/0"
    beta_results = os.listdir(beta)
    beta_results = [beta + "/" + x for x in beta_results]
    mode_3_results = [old_rosenbaum] + beta_results
    label = ["Shervani Tabar"] + ["beta={}".format(str(1/(10**i))) for i in range(0, 5, 1)]
    
    schedularT0 = "results/mode_3_all/schedularT0/0"
    schedularT0_results = os.listdir(schedularT0)
    schedularT0_results = [schedularT0 + "/" + x for x in schedularT0_results]
    mode_3_results = [old_rosenbaum] + [beta_results[2]] + schedularT0_results
    label = ["Shervani Tabar", "beta=0.1"] + ["schedularT0={}".format([1, 5, 10, 20, 30, 40][i]) for i in range(6)]

    different_y_0 = "results/different_y_0/0"
    different_y_0_results = os.listdir(different_y_0)
    different_y_0_results = [different_y_0 + "/" + x for x in different_y_0_results][-2:]
    mode_3_results = [old_rosenbaum] + different_y_0_results
    label = ["Shervani Tabar", "different_y 3 chems", "different_y 5 chems"]

    y0_individual_v_ind = "results/different_y_ind_v_diff_lr/0"
    y0_individual_v_ind_results = os.listdir(y0_individual_v_ind)
    y0_individual_v_ind_results = [y0_individual_v_ind + "/" + x for x in y0_individual_v_ind_results][-1]
    mode_3_results = [old_rosenbaum] + [different_y_0_results[0]] + [y0_individual_v_ind_results]

    label = ["Shervani Tabar", "different_y 3 chems", "layer dependent v individual"]
    
    
    runner_individual_no_bias = "results_runner/runner_individual_no_bias/0"
    runner_individual_no_bias_results = os.listdir(runner_individual_no_bias)
    runner_individual_no_bias_results = [runner_individual_no_bias + "/" + x for x in runner_individual_no_bias_results]
    mode_3_results = runner_individual_no_bias_results
    label = [10, 20, 30, 40, 50, 60, 70, 80]

    runner_diff_y0 = "results_runner/runner_diff_y0/0"
    runner_diff_y0_results = os.listdir(runner_diff_y0)
    runner_diff_y0_results = [runner_diff_y0 + "/" + x for x in runner_diff_y0_results]
    temp = runner_diff_y0_results[1]
    runner_diff_y0_results[1] = runner_diff_y0_results[2]
    runner_diff_y0_results.append(temp)
    runner_diff_y0_results.pop(2)
    mode_3_results = runner_diff_y0_results
    label = [10, 20, 30, 40, 50, 60, 80, 120]

    runner_different_y_ind_v_diff = "results_runner/runner_different_y_ind_v_diff/0"
    runner_different_y_ind_v_diff_results = os.listdir(runner_different_y_ind_v_diff)
    runner_different_y_ind_v_diff_results = [runner_different_y_ind_v_diff + "/" + x for x in runner_different_y_ind_v_diff_results]
    temp = runner_different_y_ind_v_diff_results[1]
    runner_different_y_ind_v_diff_results[1] = runner_different_y_ind_v_diff_results[2]
    runner_different_y_ind_v_diff_results.append(temp)
    runner_different_y_ind_v_diff_results.pop(2)
    mode_3_results = runner_different_y_ind_v_diff_results
    label = [10, 20, 30, 40, 50, 60, 80, 120]

    save_prefix = "runner_different_y_ind"

    
    print(mode_3_results)
    multi_plot_accuracy(mode_3_results, label, window_size=20, save_dir=save_dir, save_name=save_prefix + ".png")
    #peak_meta_accuracy_scatter_plot(mode_3_results, label, save_dir=save_dir, save_name=save_prefix + "_s.png")
    #meta_accuracy_scatter_plot(mode_3_results, label, save_dir=save_dir, save_name=save_prefix + "_sc.png", inital_cutoff=400, final_cutoff=450, window_size=10)
    #multi_plot_loss(mode_3_results, label, window_size=20, save_dir=save_dir, save_name=save_prefix + "_loss.png")
    print("Done")





    """label = ["Shervani-Tabar"] + ["{} Chemicals".format(i) for i in range(2, 6)]

    add_baselines = results_dir + "/Mode_1/baselines/0/"
    add_baselines_results = os.listdir(add_baselines)
    add_baselines_results = [add_baselines + x for x in add_baselines_results][1:5]
    add_baselines_results = [old_rosenbaum] + add_baselines_results
    
    plt.rc('font', family='serif', size=15)
    plt.figure(figsize=(10, 5))
    meta_accuracy_scatter_plot_2(add_baselines_results, label, save_dir=save_dir, save_name="add_3_error.png", inital_cutoff=140, final_cutoff=180, window_size=20, color='red', legend="Normal RCN")
    meta_accuracy_scatter_plot_2(more_individual_results, label, save_dir=save_dir, save_name="add_3_error.png", inital_cutoff=140, final_cutoff=180, window_size=20, color='blue', legend="Layer Dependent RCN")
    plt.xlabel('Model')
    plt.ylabel('Meta-Training Episode Accuracy (Average)')
    plt.title('Meta-Training Episode Accuracy (Average over {} episodes)'.format(20))
    plt.grid()
    plt.legend()
    plt.savefig(save_dir + "/" + "individual_3_error", bbox_inches="tight")

    #multi_plot_accuracy(mode_3_results, label, window_size=20, save_dir=save_dir, save_name="mode3.png")
    
    save_dir = os.getcwd() + "/graphs"
    plt.rc('font', family='serif', size=10)
    matrix_plot(None, "Layer K 2 Matrix", save_dir=save_dir)"""



    
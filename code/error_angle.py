import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    #fashion_mnist_rosenbaum = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\fashion_mnist\runner_rosenbaum_fashion\0"
    #fashion_5_chem = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\fashion_mnist\runner_mode_6_5_chem_500_tau_fashion\0"
    #fashion_feedback = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\fashion_mnist\runner_mode_6_feedback_fashion_finetuned\0"
    #folders = [fashion_mnist_rosenbaum, fashion_5_chem, fashion_feedback]

    mode_6_DFA_grad = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_mode_6_DFA_grad\0"
    runner_rosenbaum_varied = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\rosenbaum\runner_rosenbaum_varied\0"
    fixed_normalised_5_chem_800 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_5_chem_mode_6_800_2\0"

    mode_6_3_chem = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_weight_mode_6\0" # 3 chem
    mode_6_7_chem_tau_500 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_7_chem_mode_6_800_min_tau_500\0"
    mode_6_tau_1000 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_5_chem_mode_6_800_min_tau_1000\0"
    mode_6_tau_500 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_5_chem_mode_6_800_min_tau_500\0"
    


    folders = [runner_rosenbaum_varied, fixed_normalised_5_chem_800, mode_6_tau_500, mode_6_tau_1000]

    plt.rc('font', family='serif', size=12)
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
    labels = ["Rosenbaum", "RCN max tau = 100", "RCN max tau = 500", "RCN max tau = 1000"]
    colors = ["red", "blue", "fuchsia", "lime", "cyan", "purple", "orange", "black", "yellow", "green"]

    for index, i in enumerate(folders):
        all_files = os.listdir(i)
        all_files_int = [int(all_files[i] ) for i in range(len(all_files))]
        all_files_int.sort()

        all_files = [str(x) for x in all_files_int]
        all_values = []
        for j in all_files:
            if(int(j) > 700):
                continue
            directory = i + "/" + j
            z = np.loadtxt(directory + "/e_ang_meta.txt")
            average = np.mean(z, axis=0)
            all_values.append(average)
        x_axis = all_files_int[:len(all_values)]
        zero_axis = [ii[0] for ii in all_values]
        one_axis = [ii[1] for ii in all_values]
        two_axis = [ii[2] for ii in all_values]
        three_axis = [ii[3] for ii in all_values]

        ax[0, 0].plot(x_axis, zero_axis, label=labels[index], color=colors[index])
        ax[0, 1].plot(x_axis, one_axis, label=labels[index], color=colors[index])
        ax[1, 0].plot(x_axis, two_axis, label=labels[index], color=colors[index])
        ax[1, 1].plot(x_axis, three_axis, label=labels[index], color=colors[index])
    
    ax[0,0].set_title("Layer 1")
    ax[0,0].set_xlabel("Training Images Per Class")
    ax[0,0].set_ylabel("Degrees")
    ax[0,0].set_ylim(0, 99)
    ax[0,0].legend()

    ax[0,1].set_title("Layer 2")
    ax[0,1].set_xlabel("Training Images Per Class")
    ax[0,1].set_ylabel("Degrees")
    ax[0,1].set_ylim(0, 99)
    #ax[0,1].legend()

    ax[1,0].set_title("Layer 3")
    ax[1,0].set_xlabel("Training Images Per Class")
    ax[1,0].set_ylabel("Degrees")
    ax[1,0].set_ylim(0, 99)
    #ax[1,0].legend()

    ax[1,1].set_title("Layer 4")    
    ax[1,1].set_xlabel("Training Images Per Class")
    ax[1,1].set_ylabel("Degrees")
    ax[1,1].set_ylim(0, 99)
    #ax[1,1].legend()

    fig.suptitle("Angle between error signals and backprop for EMNIST Meta-trained models")
    fig.tight_layout()
    save_dir = os.getcwd() + "/graphs/"
    plt.savefig(save_dir + "max_tau_e_angle.png")
    plt.show()
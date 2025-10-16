import os

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    runner_diff_y0_3_chems = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_diff_y0_3_chems\0"
    runner_individual_no_bias = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_individual_no_bias\0"
    runner_different_y_ind_v_diff_lr = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_different_y_ind_v_diff_lr\0"
    runner_super_varied = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_super_varied\0"
    runner_y0_3_extra_long_500 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_y0_3_extra_long_500\0"
    runner_y0_3_extra_long800 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_y0_3_extra_long_800\0"
    runner_rosenbaum_50 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_rosenbaum_50\0"
    runner_rosenbaum_varied = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\rosenbaum\runner_rosenbaum_varied\0"
    runner_y0_4_extra_long_100 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_y0_4_extra_long_100\0"
    runner_y0_4_extra_long_120 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_y0_4_extra_long_120\1"
    runner_lr_5 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_lr_5\0"

    runner_4_chems = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_y0_4_extra_long_200\0"
    runner_5_chems_500_gpu = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_y0_5_extra_long_200\runner_500_epochs_gpu\0"
    runner_5_chems_500_cpu = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_y0_5_extra_long_200\runner_500_epochs_cpu\0"
    runner_post_train = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_y0_5_extra_long_200\runner_200_epochs_post_train_30_70\0"
    runner_100_post_train = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_y0_5_extra_long_200\runner_100_epochs_post_train_30_70\0"

    runner_director = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\min_tau\runner_min_tau_testing_"
    end = ["1_1_50", "2", "3", "4", "5", "20", "30", "50"]
    all_folders = []
    for i in end:
        all_folders.append(runner_director + i + r"\0")

    runner_DFA_grad_500 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_DFA_grad_test\0"
    runner_DFA_grad = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_DFA_grad_800\0"
    runner_DFA = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_DFA_test\0"
    runnner_FA_no_Grad = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_FA_No_Grad_Test\0"

    runner_eta = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_eta\0"

    runner_combined = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_Combined\0"
    runner_combined_2 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_Combined_v3\0"

    runner_rosenbaum_fashion_mnist = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_rosenbaum_fashion_mnist\0"
    runner_3chem_fashion_mnist = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_3chem_fashion_mnist\0"
    runner_3chem_fashion_mnist2 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_3chem_fashion_mnist2\0"

    backprop = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_backprop\0"

    normalised_dodge_5 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_5\runner_normalised_weight_3chem_new5a\0"
    normalised_non_dodge_5 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_5\runner_normalised_3_chem\0"
    normalised_5_chem = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_5\runner_normalised_weight_5_chem_800\0"

    fixed_normalised = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_weight_mode_6\0"  # 3 chem
    fixed_normalised_feedback = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_mode6_wtih_feedback\0"
    fixed_normalised_5_chem = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_5_chem_mode_6\0"
    fixed_normalised_5_chem_800 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_5_chem_mode_6_800_2\0"

    fashion_mnist_rosenbaum = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\fashion_mnist\runner_rosenbaum_fashion\0"
    fashion_5_chem = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\fashion_mnist\runner_mode_6_5_chem_500_tau_fashion\0"
    fashion_backprop = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\fashion_mnist\runner_backprop_fashion\0"
    fashion_feedback = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\fashion_mnist\runner_mode_6_feedback_fashion_finetuned\0"

    mode_7 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_7\runner_normalise_mode_7_FA\0"

    mode_6_DFA = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_mode_6_DFA_grad\0"
    mode_6_tau_1000 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_5_chem_mode_6_800_min_tau_1000\0"
    mode_6_tau_500 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_5_chem_mode_6_800_min_tau_500\0"
    mode_6_7_chem_tau_500 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_7_chem_mode_6_800_min_tau_500\0"
    mode_6_7_chem_tau_1000 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_7_chem_mode_6_800_min_tau_1000_2\0"
    runner_mode_6_2_chem = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_mode_6_2_chem\0"

    scalar_chem_1 = (
        os.getcwd() + r"/results_runner/scalar/runner_mode_6_1_chem_scalar/0"
    )
    scalar_chem_3 = (
        os.getcwd() + r"/results_runner/scalar/runner_mode_6_3_chem_scalar/0"
    )
    scalar_chem_5 = (
        os.getcwd() + r"/results_runner/scalar/runner_mode_6_5_chem_scalar/0"
    )
    scalar_chem_7 = (
        os.getcwd() + r"/results_runner/scalar/runner_mode_6_7_chem_scalar/0"
    )

    DFA_chem_1 = os.getcwd() + r"/results_runner/DFA/runner_mode_6_1_chem_DFA/0"
    DFA_chem_3 = os.getcwd() + r"/results_runner/DFA/runner_mode_6_3_chem_DFA/0"
    DFA_chem_5 = os.getcwd() + r"/results_runner/DFA/runner_mode_6_5_chem_DFA/0"
    DFA_chem_7 = os.getcwd() + r"/results_runner/DFA/runner_mode_6_7_chem_DFA/0"
    # print(all_folders)
    # backprop_14 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_rnn_backprop_2_layer_and_forward\14\0"
    # backprop_28 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_rnn_backprop_2_layer_and_forward\28\0"
    # backprop_56 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_rnn_backprop_2_layer_and_forward\56\0"
    # backprop_112 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_rnn_backprop_2_layer_and_forward\112\0"
    # backprop_392 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_rnn_backprop_2_layer_and_forward\392\0"
    # backprop_784 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_rnn_backprop_2_layer_and_forward\784\0"

    # backprop_14_4 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_rnn_backprop_bio\14\0"
    # backprop_28_4 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_rnn_backprop_bio\28\0"
    # backprop_56_4 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_rnn_backprop_bio\56\0"
    # backprop_112_4 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_rnn_backprop_bio\112\0"

    small_examples = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_mode_6_5_train_1500_small_examples\0"
    small_examples_DFA_grad = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_mode_6_5_train_1500_DFA_grad_small_examples\0"

    runner_mode_4_ind_ind_v = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_mode_4_ind_ind_v\0"
    runner_mode_6_ind_non_ind_v = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\runner_mode_6_ind_non_ind_v\0"

    # runner_DFA_grad_fashion = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\fashion_mnist\runner_mode_6_5_train_DFA_grad_fashion\0"
    runner_mode_4_1_chems = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_mode_4_1_chems\0"
    runner_mode_6_1_chem = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_mode_6_1_chem\0"

    scalar_folders = [
        scalar_chem_1,
        scalar_chem_3,
        scalar_chem_5,
        scalar_chem_7,
    ]
    DFA_folders = [
        DFA_chem_1,
        DFA_chem_3,
        DFA_chem_5,
        DFA_chem_7,
    ]

    Baseline = backprop
    # runner_folders = all_folders
    labels = [
        "Backprop",
        "FA",
        "DFA",
        "Scalar",
    ]
    # labels = ["1.5", "2", "3", "4", "5", "20", "30", "50"]
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

    limit = 130
    i = backprop
    all_files = os.listdir(backprop)
    all_files_int = [int(all_files[i]) for i in range(len(all_files))]
    all_files_int.sort()
    for k in range(len(all_files_int)):
        if all_files_int[k] > limit:
            all_files_int = all_files_int[:k]
            break

    all_files = [str(x) for x in all_files_int]
    all_values = np.array([])
    for j in all_files:
        directory = i + "/" + j
        z = np.loadtxt(directory + "/acc_meta.txt")
        average = np.mean(z, axis=0)
        # median = np.median(z, axis=0)
        all_values = np.append(all_values, average)
    all_values = all_values[:limit]
    backprop_x_axis = all_files_int[: len(all_values)]
    backprop_values = all_values

    print("Baseline (Backprop) mean: " + str(np.mean(backprop_values)))
    scalar_means = []
    for index, i in enumerate(scalar_folders):
        all_files = os.listdir(i)
        all_files_int = [int(all_files[i]) for i in range(len(all_files))]
        all_files_int.sort()
        for k in range(len(all_files_int)):
            if all_files_int[k] > limit:
                all_files_int = all_files_int[:k]
                break

        all_files = [str(x) for x in all_files_int]
        all_values = np.array([])
        index_2 = 0
        for j in backprop_x_axis:
            j_str = str(j)
            directory = i + "/" + j_str
            print(directory)
            z = np.loadtxt(directory + "/acc_meta.txt")
            average = np.mean(z, axis=0)
            adjusted_average = average
            index_2 += 1
            # median = np.median(z, axis=0)
            all_values = np.append(all_values, average)
        all_values = all_values[:limit]
        all_values = np.array(all_values)
        all_values_mean = np.mean(all_values) - np.mean(backprop_values)
        scalar_means.append(all_values_mean)

    DFA_means = []
    for index, i in enumerate(DFA_folders):
        all_files = os.listdir(i)
        all_files_int = [int(all_files[i]) for i in range(len(all_files))]
        all_files_int.sort()
        for k in range(len(all_files_int)):
            if all_files_int[k] > limit:
                all_files_int = all_files_int[:k]
                break

        all_files = [str(x) for x in all_files_int]
        all_values = np.array([])
        index_2 = 0
        for j in backprop_x_axis:
            j_str = str(j)
            directory = i + "/" + j_str
            print(directory)
            z = np.loadtxt(directory + "/acc_meta.txt")
            average = np.mean(z, axis=0)
            adjusted_average = average
            index_2 += 1
            # median = np.median(z, axis=0)
            all_values = np.append(all_values, average)
        all_values = all_values[:limit]
        all_values = np.array(all_values)
        all_values_mean = np.mean(all_values) - np.mean(backprop_values)
        DFA_means.append(all_values_mean)

    plt.rc("font", family="serif", size=14)
    plt.figure(figsize=(13, 8))
    x_axis = np.array([1, 3, 5, 7])
    plt.plot(x_axis, scalar_means, label="Scalar", color="blue", marker="o")
    plt.plot(x_axis, DFA_means, label="DFA", color="fuchsia", marker="o")
    plt.axhline(y=0, color="black", linestyle="--")
    plt.xticks(x_axis)
    plt.xlabel("Number of Chemicals")
    plt.ylabel("Evaluation Accuracy")
    plt.tight_layout()
    plt.show()

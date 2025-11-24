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

    backprop = os.getcwd() + r"\results_runner\runner_backprop\0"

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

    new_scalar_5 = os.getcwd() + r"/results_runner/new_scalar/runner_scalar_fixed_5_4/0"
    new_scalar_1 = os.getcwd() + r"/results_runner/new_scalar/runner_scalar_fixed_1/0"
    new_scalar_3_3 = (
        os.getcwd() + r"/results_runner/new_scalar/runner_scalar_fixed_3_3/0"
    )
    new_scalar_3 = os.getcwd() + r"/results_runner/new_scalar/runner_scalar_fixed_3_6/0"
    new_scalar_3_mode_9 = (
        os.getcwd() + r"/results_runner/new_scalar/runner_mode_9_scalar_3_chems/0"
    )
    runner_mode_9_scalar_GPU_5 = (
        os.getcwd() + r"/results_runner/new_scalar/runner_mode_9_scalar_GPU_5/0"
    )
    runner_mode_9_scalar_fixed = (
        os.getcwd() + r"/results_runner/new_scalar/runner_mode_9_fixed/0"
    )
    runner_mode_9_scalar_fixed_8 = (
        os.getcwd() + r"/results_runner/new_scalar/runner_mode_9_fixed_8/0"
    )
    runner_mode_9_scalar_GPU_6 = (
        os.getcwd() + r"/results_runner/new_scalar/runner_mode_9_scalar_GPU_6/0"
    )
    runner_mode_9_5_all_ones = (
        os.getcwd() + r"/results_runner/new_scalar/runner_mode_9_scalar_all_ones_3/0"
    )
    runner_mode_9_5_all_ones_diff = (
        os.getcwd() + r"/results_runner/new_scalar/runner_mode_9_5_all_ones_diff/0"
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

    # rnns:
    rnn_backprop_mode_1 = (
        os.getcwd() + r"/rnn_results_runner/runner_rnn_backprop_mode_1/112/0"
    )
    rnn_backprop_mode_4 = (
        os.getcwd() + r"/rnn_results_runner/runner_rnn_backprop_mode_4_128_3/112/0"
    )
    rnn_fast_mode_1 = os.getcwd() + r"/rnn_results_runner/runner_rnn_fast_mode_1/0"
    rnn_fast_mode_3 = os.getcwd() + r"/rnn_results_runner/runner_rnn_fast_mode_3/0"
    rnn_fast_mode_4 = (
        os.getcwd() + r"/rnn_results_runner/runner_rnn_fast_mode_4_128_2/0"
    )
    rnn_fast_mode_4_2 = (
        os.getcwd() + r"/rnn_results_runner/runner_rnn_fast_mode_4_128_longer/0"
    )
    runner_rnn_fast_mode_4_128_longer_test_time_compute = (
        os.getcwd()
        + r"/rnn_results_runner/runner_rnn_fast_mode_4_128_longer_test_time_compute/0"
    )

    rflo_mode_4 = (
        os.getcwd() + r"/rnn_results_runner/runner_test_rnn_rflo_linear_2/112/0"
    )
    rflo_weird_mode_4 = (
        os.getcwd() + r"/rnn_results_runner/runner_rnn_rflo_mode_4_3/112/0"
    )

    more_layers_6_DFA_5 = (
        os.getcwd() + r"/results_runner/more_layers/runner_DFA_6_layer_true_3_5_chem/0"
    )
    more_layers_10_DFA_5 = (
        os.getcwd() + r"/results_runner/more_layers/runner_DFA_10_layer_5_chem/0"
    )
    backprop_more_layers_10 = (
        os.getcwd() + r"/results_runner/runner_backprop_10_layer_EMNIST_3/0/"
    )

    runner_DFA_5_layer_5_chem_fashion_new = (
        os.getcwd() + r"/results_runner/DFA/runner_DFA_5_layer_5_chem_fashion_new/0"
    )

    runner_DFA_10_layer_5_chem_fashion_new = (
        os.getcwd() + r"/results_runner/DFA/runner_DFA_10_layer_5_chem_fashion_new/0"
    )

    runner_backprop_10_layer_fashion_mnist = (
        os.getcwd() + r"/results_runner/runner_backprop_10_layer_fashion_mnist/0/"
    )

    runner_mode_9 = os.getcwd() + r"/results_runner/runner_mode_9_rand/0/"
    runner_mode_9_5_all_ones_DFA = (
        os.getcwd() + r"/results_runner/DFA/runner_mode_9_5_all_ones_DFA/0/"
    )
    runner_mode_6_scalar_NAO_7 = (
        os.getcwd() + r"/results_runner/new_scalar/runner_mode_6_scalar_NAO_7/0/"
    )
    runner_mode_6_scalar_500_3 = (
        os.getcwd() + r"/results_runner/new_scalar/runner_mode_6_scalar_500_3/0/"
    )
    runner_mode_6_scalar_all_ones = (
        os.getcwd() + r"/results_runner/new_scalar/runner_mode_6_scalar_all_ones/0/"
    )
    runner_mode_6_scalar_10_same = (
        os.getcwd() + r"/results_runner/new_scalar/runner_mode_6_scalar_10_same/0/"
    )
    runner_mode_9_scalar_10_same = (
        os.getcwd() + r"/results_runner/new_scalar/runner_mode_9_scalar_10_same/0/"
    )
    runner_mode_9_scalar_10_same_2 = (
        os.getcwd() + r"/results_runner/new_scalar/runner_mode_9_scalar_10_same_2/0/"
    )
    runner_mode_9_scalar_10_same_3 = (
        os.getcwd() + r"/results_runner/new_scalar/runner_mode_9_scalar_10_same_3/0/"
    )
    """runner_folders = [
        rnn_backprop_mode_4,
        rnn_fast_mode_4_2,
        rflo_mode_4,
        # rflo_weird_mode_4,
    ]"""
    """runner_folders = [
        scalar_chem_1,
        scalar_chem_3,
        scalar_chem_5,
        scalar_chem_7,
    ]"""
    runner_folders = [
        backprop,
        # DFA_chem_1,
        # DFA_chem_3,
        # backprop,
        # small_examples,
        # scalar_chem_5,
        # DFA_chem_5,
        # runner_mode_9,
        # runner_mode_9_5_all_ones_DFA,
        # new_scalar_5,
        # new_scalar_5,
        runner_mode_6_scalar_10_same,
        # DFA_chem_7,
        # runner_rosenbaum_varied,
        # new_scalar_1,
        # new_scalar_3,
        # new_scalar_3_mode_9,
        # new_scalar_5,
        runner_mode_9_scalar_10_same_2,
        # runner_mode_9_scalar_10_same_3,
        new_scalar_1,
        # runner_mode_9_scalar_fixed_8,
    ]
    """runner_folders = [
        backprop,
        new_scalar_1,
        new_scalar_3,
        new_scalar_5,
    ]"""
    """runner_folders = [
        backprop,
        backprop_more_layers_10,
        DFA_chem_5,
        more_layers_10_DFA_5,
    ]"""
    """runner_folders = [
        fashion_backprop,
        runner_backprop_10_layer_fashion_mnist,
        runner_DFA_5_layer_5_chem_fashion_new,
        runner_DFA_10_layer_5_chem_fashion_new,
        fashion_mnist_rosenbaum,
        fashion_5_chem,
    ]"""

    """runner_folders = [
        backprop,
        rnn_fast_mode_4_2,
        rflo_mode_4,
    ]"""
    """runner_folders = [
        fashion_backprop,
        fashion_mnist_rosenbaum,
        fashion_5_chem,
        runner_3chem_fashion_mnist2,
        runner_3chem_fashion_mnist,
    ]"""
    # runner_folders = all_folders
    labels = [
        "Backprop",
        # "Simple Synapse Direct Vector-Error Feedback",  # direct vector-error feedback
        # "1 Chem DFA",
        # "3 State",
        # "5 State",
        # "7 State",
        # "Layerwise Vector-Error Feedback (FA)",
        # "Scalar",
        # "Direct Vector-Error Feedback (DFA)",
        # "Direct Scalar-Error Feedback (scalar)",
        # "$W$ and $H$ Normalised Independently - $z$ all ones",
        "$W$ and $H$ Normalised Independently",
        "$H$ Normalised using $W$",
        # "$H$ Normalised using $W$",
        "1 Chem",
    ]
    """labels = [
        "BPTT",
        "Complex Synapse",
        "RFLO",
    ]"""
    """labels = [
        "Backprop",
        "Simple Synapse",  # direct vector-error feedback
        "3 State Variables",
        "3 State Variables New",
        "5 State Variables",
        "5 State Variables New",
        # "7 State Variables",
    ]"""
    """labels = [
        "Backprop 5 layers",
        "Backprop 10 layers",
        "5 Layers",
        "10 Layers",
        "rose",
        "pre",
    ]"""
    # labels = ["1.5", "2", "3", "4", "5", "20", "30", "50"]
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
    plt.figure(figsize=(9, 5))
    limit = 160
    for index, i in enumerate(runner_folders):
        all_files = os.listdir(i)
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
            std = np.std(z, axis=0)
            # median = np.median(z, axis=0)
            all_values = np.append(all_values, average)
            # print(i, j, average, std)
        all_values = all_values[:limit]
        x_axis = all_files_int[: len(all_values)]
        """if index < 2:
            plt.plot(
                x_axis,
                all_values,
                # label=labels[index],
                color=colors[index],
                linestyle="--",
            )
            plt.scatter(x_axis, all_values, color=colors[index])
        else:
            plt.plot(x_axis, all_values, label=labels[index], color=colors[index - 2])
            plt.scatter(x_axis, all_values, color=colors[index - 2])"""
        plt.plot(
            x_axis,
            all_values,
            label=labels[index],
            color=colors[index],
        )
        plt.scatter(x_axis, all_values, color=colors[index])
        # plt.fill_between(x_axis, 0, all_values, alpha=0.5, facecolor=colors[index])
        x_axis = np.array(x_axis)  # type: ignore

    # plt.title(r"Scalar Error Complex Synapse Performance")
    plt.xlabel("Training Samples Per Class")
    # plt.axvline(x=5, color='blue', linestyle='--', label="Min Training Images for Small Examples")
    plt.axvline(x=5, color="black", linestyle="--")
    plt.axvline(x=80, color="black", linestyle="--")
    plt.legend()
    plt.xlim(-5, limit + 10)
    plt.ylim(0, 1)
    plt.ylabel("Evaluation Accuracy")
    plt.tight_layout()
    save_dir = os.getcwd() + "/rnns_graphs/"
    plt.savefig(save_dir + "scalar_mode_comparison.png")
    plt.show()

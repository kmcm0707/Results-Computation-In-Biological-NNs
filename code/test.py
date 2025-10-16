import os

import matplotlib.colors as mat_colors
import matplotlib.pyplot as plt
import numpy as np
import torch


def make_irregular_grid(
    figsize=(18, 13),
    row_heights=(1.0, 1.0, 1.0),  # relative heights for rows 1–3
    wspace=0.1,  # horizontal space between panels within a row
    hspace=0.12,  # vertical space between rows
    row1_equal_cols=3,  # number of equal-width panels in row 1 (default 3)
    # Row 3 proportions: [small, big, big] where 'small' is a fraction of 'big'
    row3_small_to_big=0.5,
    # Row 2 proportions: [small, big, right_block], where small and big are like row 3,
    # and right_block is sized relative to 'big' (e.g., 1.0 means equal to big)
    row2_small_to_big=0.5,
    row2_right_block_vs_big=1.0,
    # Spacing inside the 2×2 block on row 2 (right side)
    row2_block_wspace=0.05,
    row2_block_hspace=0.05,
):
    """
    Returns (fig, axes_dict) where axes_dict contains:
      - 'row1': [ax1, ax2, ax3]
      - 'row2': {'left': ax, 'middle': ax, 'block_2x2': [a,b,c,d]}
      - 'row3': [ax_small, ax_big1, ax_big2]

    Proportion logic:
      * Row 1: equal columns
      * Row 3: width_ratios = [row3_small_to_big, 1, 1]
      * Row 2: width_ratios = [row2_small_to_big, 1, row2_right_block_vs_big]
    """
    fig = plt.figure(figsize=figsize)
    # Top-level gridspec: 3 rows, one column, with custom row heights
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=row_heights, hspace=hspace)

    axes = {}

    # -------- Row 1: three equal-width panels --------
    gs1 = gs[0].subgridspec(1, row1_equal_cols, wspace=wspace * 0.5)
    row1_axes = [fig.add_subplot(gs1[0, i]) for i in range(row1_equal_cols)]
    axes["row1"] = row1_axes

    # -------- Row 2: [small, big, right-block(2x2)] --------
    # Width ratios are relative; Matplotlib normalizes them automatically
    gs2 = gs[1].subgridspec(
        1,
        3,
        wspace=wspace,
        width_ratios=[row2_small_to_big, row2_right_block_vs_big, 1.0],
    )
    gs2_block_left = gs2[0, 0].subgridspec(
        1, 2, wspace=row2_block_wspace, width_ratios=[1, 0.14]
    )
    ax2_left = [
        fig.add_subplot(gs2_block_left[0, 0]),
        fig.add_subplot(gs2_block_left[0, 1]),
    ]
    ax2_mid = fig.add_subplot(gs2[0, 2])

    # Right block is a 2×2 grid of small equal panels
    gs2_block = gs2[0, 1].subgridspec(
        1, 4, wspace=row2_block_wspace, hspace=row2_block_hspace
    )
    ax2_block = [
        fig.add_subplot(gs2_block[0, 0]),
        fig.add_subplot(gs2_block[0, 1]),
        fig.add_subplot(gs2_block[0, 2]),
        fig.add_subplot(gs2_block[0, 3]),
    ]
    axes["row2"] = {"left": ax2_left, "middle": ax2_mid, "block_2x2": ax2_block}

    # -------- Row 3: [small, big, big] --------
    gs3 = gs[2].subgridspec(
        1, 4, wspace=wspace, width_ratios=[row3_small_to_big, 0.66, 0.66, 0.66]
    )
    row3_axes = [
        fig.add_subplot(gs3[0, 0]),
        fig.add_subplot(gs3[0, 1]),
        fig.add_subplot(gs3[0, 2]),
        fig.add_subplot(gs3[0, 3]),
    ]
    axes["row3"] = row3_axes

    return fig, axes


# ---------- Example usage ----------
if __name__ == "__main__":
    fig, axes = make_irregular_grid(
        figsize=(21, 11),
        row_heights=(1.0, 1.3, 1.0),
        wspace=0.2,
        hspace=0.3,
        row3_small_to_big=0.5,  # Row 3 small = 0.5 × big
        row2_small_to_big=0.8,  # Row 2 small = 0.5 × big
        row2_right_block_vs_big=2.3,  # Right 2×2 block width = 1.0 × big
        row2_block_wspace=0.2,
        row2_block_hspace=0.3,
    )

    axes["row1"][0].axis("off")
    axes["row1"][1].axis("off")

    results_dir = os.getcwd() + "/results"
    save_dir = os.getcwd() + "/rnns_graphs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    five_chem = (
        results_dir + r"/DFA_5_regularized_not_K/0/20251009-180024/UpdateWeights.pth"
    )
    five_chem_P_matrix = torch.load(
        five_chem, map_location=torch.device("cpu"), weights_only=True
    )["P_matrix"]

    five_chem_P_matrix = np.array(five_chem_P_matrix, dtype=np.float32, copy=True)
    # remove columns 5
    five_chem_P_matrix = np.delete(five_chem_P_matrix, np.s_[5], axis=1)
    # remove column 7
    five_chem_P_matrix = np.delete(five_chem_P_matrix, np.s_[7], axis=1)
    # remove column 7
    five_chem_P_matrix = np.delete(five_chem_P_matrix, np.s_[6], axis=1)
    divnorm = mat_colors.TwoSlopeNorm(vmin=-0.012, vcenter=0.0, vmax=0.012)
    axes["row2"]["left"][0].matshow(five_chem_P_matrix, cmap="seismic", norm=divnorm)
    """fig.colorbar(
        axes["row2"]["left"].images[0],
        ax=axes["row2"]["left"],
        orientation="vertical",
        fraction=0.03,
    )"""
    # axes["row2"]["left"].colorbar(norm=divnorm, cmap="seismic")
    """axes["row2"]["left"][0].set_title(
        "P Matrix for 5 State Complex Synapse",
        fontdict={"fontsize": 10},
    )"""
    axes["row2"]["left"][0].xaxis.set_ticks_position("bottom")
    axes["row2"]["left"][0].xaxis.set_label_position("bottom")
    axes["row2"]["left"][0].set_xlabel("Plasticity Motif")
    axes["row2"]["left"][0].set_ylabel("Complex Synapse State Variables")
    axes["row2"]["left"][0].set_xticks(
        ticks=np.arange(0, 7),
        labels=["$m_1$", "$m_2$", "$m_3$", "$m_4$", "$m_5$", "$m_6$", "$m_7$"],
    )
    axes["row2"]["left"][0].set_yticks(
        ticks=np.arange(0, 5),
        labels=np.arange(1, 6),
    )

    five_chem = (
        results_dir + r"/DFA_5_regularized_not_K/0/20251009-180024/UpdateWeights.pth"
    )
    five_chem_v_vector = torch.load(
        five_chem, map_location=torch.device("cpu"), weights_only=True
    )["v_vector"]

    five_chem_v_vector = torch.softmax(five_chem_v_vector, dim=1)

    five_chem_v_vector = np.array(five_chem_v_vector.T, dtype=np.float32, copy=True)

    divnorm = mat_colors.TwoSlopeNorm(vmin=-0.012, vcenter=0.0, vmax=0.3)

    axes["row2"]["left"][1].matshow(five_chem_v_vector, cmap="seismic", norm=divnorm)
    axes["row2"]["left"][1].set_yticks(np.arange(-0.5, 5, 1), minor=True)
    axes["row2"]["left"][1].grid(
        which="minor", color="black", linestyle="-", linewidth=1
    )
    """axes["row2"]["left"][1].set_xticks(
        ticks=np.arange(0, 1),
        labels=["$\mathbf{\pi}$"],
    )"""
    """axes["row2"]["left"][1].set_title(
        "V vector for 5 State Complex Synapse",
        fontdict={"fontsize": 10},
    )"""
    axes["row2"]["left"][1].set_xticks([], [])
    axes["row2"]["left"][1].set_yticks([], [])
    # axes["row2"]["left"][1].set_xlabel("Chemical Index")

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

    new_scalar_5 = os.getcwd() + r"/results_runner/new_scalar/runner_scalar_fixed_5_3/0"

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
    rflo_murray = os.getcwd() + r"/rnn_results_runner/runner_murray_rflo_5/112/0"

    new_scalar_5 = os.getcwd() + r"/results_runner/new_scalar/runner_scalar_fixed_5_4/0"
    new_scalar_1 = os.getcwd() + r"/results_runner/new_scalar/runner_scalar_fixed_1/0"
    new_scalar_3_3 = (
        os.getcwd() + r"/results_runner/new_scalar/runner_scalar_fixed_3_3/0"
    )
    new_scalar_3 = os.getcwd() + r"/results_runner/new_scalar/runner_scalar_fixed_3_6/0"

    runner_folders = [
        backprop,
        # small_examples,
        # amples,
        # scalar_chem_5,
        small_examples,
        DFA_chem_5,
        new_scalar_5,
        runner_rosenbaum_varied,
        # scalar_chem_5,
    ]
    # runner_folders = all_folders
    labels = [
        "backprop (ADAM optimizer)",
        # "1 Chem DFA",
        # "3 State",
        # "5 State",
        # "7 State",
        # "Layerwise Vector-Error Feedback (FA)",
        # "Scalar",
        "CS ($D_h=5$) w/ layerwise vector-error feedback (FA)",
        "CS ($D_h=5$) w/ direct vector-error feedback (DFA)",
        "CS ($D_h=5$) w/ direct scalar-error feedback (DSEF)",
        "Shervani-Tabar and Rosenbaum (2023)",
    ]

    colors = [
        "#0343df",
        "#e50000",
        "#f97306",
        "#7e1e9c",
        "#008000",
        "#800080",
        "#ffa500",
        "#000000",
        "#ffff00",
        "#008000",
    ]

    limit = 130
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
            # median = np.median(z, axis=0)
            all_values = np.append(all_values, average)
        all_values = all_values[:limit]
        x_axis = all_files_int[: len(all_values)]
        axes["row1"][2].plot(
            x_axis, all_values, label=labels[index], color=colors[index]
        )
        # plt.fill_between(x_axis, 0, all_values, alpha=0.5, facecolor=colors[index])
        axes["row1"][2].scatter(x_axis, all_values, color=colors[index], s=10)
        x_axis = np.array(x_axis)

    # axes["row1"][2].set_title(r"Error Feedback Comparison on EMNIST")
    axes["row1"][2].set_xlabel("training samples per class")
    # plt.axvline(x=5, color='blue', linestyle='--', label="Min Training Images for Small Examples")
    axes["row1"][2].axvline(
        x=5,
        color="black",
        linestyle="--",  # label="Min / Max Training Images"
    )
    axes["row1"][2].axvline(x=80, color="black", linestyle="--")
    axes["row1"][2].legend()
    axes["row1"][2].set_xlim(-5, limit + 10)
    axes["row1"][2].set_ylim(0, 1)
    axes["row1"][2].set_ylabel("validation accuracy")

    runner_folders = [
        backprop,
        new_scalar_1,
        new_scalar_3,
        new_scalar_5,
    ]

    labels = [
        "backprop",
        "CS $(D_h=1)$ w/ DSEF",  # direct vector-error feedback
        "CS $(D_h=3)$ w/ DSEF",  # 3 state variables
        "CS $(D_h=5)$ w/ DSEF",  # 5 state variables
        # "CS $(D_h=7)$",  # 7 state variables
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

    limit = 130
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
            # median = np.median(z, axis=0)
            all_values = np.append(all_values, average)
        all_values = all_values[:limit]
        x_axis = all_files_int[: len(all_values)]
        axes["row2"]["middle"].plot(
            x_axis, all_values, label=labels[index], color=colors[index]
        )
        # plt.fill_between(x_axis, 0, all_values, alpha=0.5, facecolor=colors[index])
        axes["row2"]["middle"].scatter(x_axis, all_values, color=colors[index], s=10)
        x_axis = np.array(x_axis)

    # axes["row2"]["middle"].set_title(
    #    r"Scalar Error State Variable Comparison on EMNIST"
    # )
    axes["row2"]["middle"].set_xlabel("training samples per class")
    # plt.axvline(x=5, color='blue', linestyle='--', label="Min Training Images for Small Examples")
    axes["row2"]["middle"].axvline(
        x=5,
        color="black",
        linestyle="--",  # label="Min / Max Training Images"
    )
    axes["row2"]["middle"].axvline(x=80, color="black", linestyle="--")
    axes["row2"]["middle"].legend()
    axes["row2"]["middle"].set_xlim(-5, limit + 10)
    axes["row2"]["middle"].set_ylim(0, 1)
    axes["row2"]["middle"].set_ylabel("validation accuracy")

    runnner_FA_no_Grad = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_FA_No_Grad_Test\0"
    runner_DFA_grad = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_DFA_grad_800\0"
    runner_combined = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_Combined\0"
    runner_DFA = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_DFA_test\0"

    mode_6_DFA_grad = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_mode_6_DFA_grad\0"
    runner_rosenbaum_varied = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\rosenbaum\runner_rosenbaum_varied\0"
    fixed_normalised_5_chem_800 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_5_chem_mode_6_800_2\0"

    mode_6_3_chem = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_weight_mode_6\0"  # 3 chem
    mode_6_7_chem_tau_500 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_7_chem_mode_6_800_min_tau_500\0"
    mode_6_tau_1000 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_5_chem_mode_6_800_min_tau_1000\0"
    mode_6_tau_500 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_5_chem_mode_6_800_min_tau_500\0"

    fixed_normalised = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_weight_mode_6\0"  # 3 chem
    fixed_normalised_feedback = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_6\runner_mode6_wtih_feedback\0"

    runner_lr_5 = r"C:\Users\kmc07\Results-Computation-In-Biological-NNs\results_runner\mode_4\runner_lr_5\0"

    # folders = [runner_rosenbaum_varied, fixed_normalised_5_chem_800, mode_6_tau_500, mode_6_tau_1000]
    new_scalar_5 = os.getcwd() + r"/results_runner/new_scalar/runner_scalar_fixed_5_3/0"
    folders = [runner_rosenbaum_varied, fixed_normalised, mode_6_DFA_grad, new_scalar_5]
    labels = [
        r"FA ($D_h=1$)",
        r"FA ($D_h=5$)",
        r"DFA ($D_h=5$)",
        r"DSEF ($D_h=5$)",
    ]

    for index, i in enumerate(folders):
        all_files = os.listdir(i)
        all_files_int = [int(all_files[i]) for i in range(len(all_files))]
        all_files_int.sort()

        all_files = [str(x) for x in all_files_int]
        all_values = []
        for j in all_files:
            if int(j) > 250:
                continue
            directory = i + "/" + j
            z = np.loadtxt(directory + "/e_ang_meta.txt")

            average = np.mean(z, axis=0)
            all_values.append(average)
        x_axis = all_files_int[: len(all_values)]
        zero_axis = [ii[0] for ii in all_values]
        one_axis = [ii[1] for ii in all_values]
        two_axis = [ii[2] for ii in all_values]
        three_axis = [ii[3] for ii in all_values]

        axes["row2"]["block_2x2"][0].plot(
            x_axis, zero_axis, label=labels[index], color=colors[index]
        )
        axes["row2"]["block_2x2"][0].scatter(
            x_axis, zero_axis, color=colors[index], s=10
        )
        axes["row2"]["block_2x2"][1].plot(
            x_axis, one_axis, label=labels[index], color=colors[index]
        )
        axes["row2"]["block_2x2"][1].scatter(
            x_axis, one_axis, color=colors[index], s=10
        )
        axes["row2"]["block_2x2"][2].plot(
            x_axis, two_axis, label=labels[index], color=colors[index]
        )
        axes["row2"]["block_2x2"][2].scatter(
            x_axis, two_axis, color=colors[index], s=10
        )
        axes["row2"]["block_2x2"][3].plot(
            x_axis, three_axis, label=labels[index], color=colors[index]
        )
        axes["row2"]["block_2x2"][3].scatter(
            x_axis, three_axis, color=colors[index], s=10
        )

    axes["row2"]["block_2x2"][0].set_title("Layer 1")
    axes["row2"]["block_2x2"][0].set_ylabel("alignment angle (degrees)")
    axes["row2"]["block_2x2"][0].set_ylim(0, 99)
    axes["row2"]["block_2x2"][0].set_yticks(np.arange(0, 100, 10))

    axes["row2"]["block_2x2"][0].legend()

    axes["row2"]["block_2x2"][1].set_title("Layer 2")
    # ax[1].set_ylabel("Degrees")
    axes["row2"]["block_2x2"][1].set_ylim(0, 99)
    # ax[0,1].legend()

    axes["row2"]["block_2x2"][2].set_title("Layer 3")
    axes["row2"]["block_2x2"][2].set_xlabel("training samples per class")
    # ax[2].set_ylabel("Degrees")
    axes["row2"]["block_2x2"][2].set_ylim(0, 99)
    # ax[1,0].legend()

    axes["row2"]["block_2x2"][3].set_title("Layer 4")
    # ax[3].set_ylabel("Degrees")
    axes["row2"]["block_2x2"][3].set_ylim(0, 99)
    # ax[1,1].legend()

    five_chem = (
        results_dir + r"/DFA_5_regularized_not_K/0/20251009-180024/UpdateWeights.pth"
    )
    five_chem_K_matrix = torch.load(
        five_chem, map_location=torch.device("cpu"), weights_only=True
    )["K_matrix"]

    five_chem_K_matrix = np.array(five_chem_K_matrix, dtype=np.float32, copy=True)
    divnorm = mat_colors.TwoSlopeNorm(vmin=-0.02, vcenter=0.0, vmax=0.02)

    axes["row3"][0].matshow(five_chem_K_matrix, cmap="seismic", norm=divnorm)
    """fig.colorbar(
        axes["row3"][0].images[0],
        ax=axes["row3"][0],
        orientation="vertical",
        fraction=0.03,
    )"""
    axes["row3"][0].xaxis.set_ticks_position("bottom")
    axes["row3"][0].xaxis.set_label_position("bottom")
    axes["row3"][0].set_xlabel("Complex Synapse State Variables")
    axes["row3"][0].set_ylabel("Complex Synapse State Variables")
    axes["row3"][0].set_yticks(
        ticks=np.arange(0, 5),
        labels=np.arange(1, 6),
    )
    axes["row3"][0].set_xticks(
        ticks=np.arange(0, 5),
        labels=np.arange(1, 6),
    )
    """axes["row3"][0].set_title(
        "K Matrix for 5 State Complex Synapse", fontdict={"fontsize": 10}
    )"""

    runner_DFA_5_layer_5_chem_fashion_new = (
        os.getcwd() + r"/results_runner/DFA/runner_DFA_5_layer_5_chem_fashion_new/0"
    )

    runner_DFA_10_layer_5_chem_fashion_new = (
        os.getcwd() + r"/results_runner/DFA/runner_DFA_10_layer_5_chem_fashion_new/0"
    )

    runner_backprop_10_layer_fashion_mnist = (
        os.getcwd() + r"/results_runner/runner_backprop_10_layer_fashion_mnist/0/"
    )

    more_layers_10_DFA_5 = (
        os.getcwd() + r"/results_runner/more_layers/runner_DFA_10_layer_5_chem/0"
    )
    backprop_more_layers_10 = (
        os.getcwd() + r"/results_runner/runner_backprop_10_layer_EMNIST_3/0/"
    )

    runner_folders = [
        backprop,
        backprop_more_layers_10,
        DFA_chem_5,
        more_layers_10_DFA_5,
    ]

    labels = ["Backprop 5 layers", "Backprop 10 layers", "5-Layer FFN", "10-Layer FFN"]

    limit = 130
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
            # median = np.median(z, axis=0)
            all_values = np.append(all_values, average)
        all_values = all_values[:limit]
        x_axis = all_files_int[: len(all_values)]
        if index < 2:
            axes["row3"][1].plot(
                x_axis,
                all_values,
                # label=labels[index],
                color=colors[index],
                linestyle="--",
            )
            axes["row3"][1].scatter(x_axis, all_values, color=colors[index], s=10)
        else:
            axes["row3"][1].plot(
                x_axis, all_values, label=labels[index], color=colors[index - 2]
            )
            axes["row3"][1].scatter(x_axis, all_values, color=colors[index - 2], s=10)
        x_axis = np.array(x_axis)

    # axes["row3"][1].set_title(
    #    r"Evaluation on Fashion-MNIST for EMNIST Meta-Trained Models"
    # )
    axes["row3"][1].set_xlabel("training samples per class")
    # plt.axvline(x=5, color='blue', linestyle='--', label="Min Training Images for Small Examples")
    axes["row3"][1].axvline(
        x=30,
        color="black",
        linestyle="--",  # label="Min / Max Training Images"
    )
    axes["row3"][1].axvline(x=80, color="black", linestyle="--")
    axes["row3"][1].legend()
    axes["row3"][1].set_xlim(-5, limit + 10)
    axes["row3"][1].set_ylim(0, 1)
    axes["row3"][1].set_ylabel("validation accuracy")

    runner_folders = [
        fashion_backprop,
        runner_backprop_10_layer_fashion_mnist,
        runner_DFA_5_layer_5_chem_fashion_new,
        runner_DFA_10_layer_5_chem_fashion_new,
    ]

    labels = ["Backprop 5 layers", "Backprop 10 layers", "5-Layer FFN", "10-Layer FFN"]

    limit = 700
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
            # median = np.median(z, axis=0)
            all_values = np.append(all_values, average)
        all_values = all_values[:limit]
        x_axis = all_files_int[: len(all_values)]
        if index < 2:
            axes["row3"][2].plot(
                x_axis,
                all_values,
                # label=labels[index],
                color=colors[index],
                linestyle="--",
            )
            axes["row3"][2].scatter(x_axis, all_values, color=colors[index], s=10)
        else:
            axes["row3"][2].plot(
                x_axis, all_values, label=labels[index], color=colors[index - 2]
            )
            axes["row3"][2].scatter(x_axis, all_values, color=colors[index - 2], s=10)
        x_axis = np.array(x_axis)

    # axes["row3"][1].set_title(
    #    r"Evaluation on Fashion-MNIST for EMNIST Meta-Trained Models"
    # )
    axes["row3"][2].set_xlabel("training samples per class")
    # plt.axvline(x=5, color='blue', linestyle='--', label="Min Training Images for Small Examples")
    axes["row3"][2].axvline(
        x=30,
        color="black",
        linestyle="--",  # label="Min / Max Training Images"
    )
    axes["row3"][2].axvline(x=80, color="black", linestyle="--")
    axes["row3"][2].legend(loc="lower right")
    axes["row3"][2].set_xlim(-5, limit + 10)
    axes["row3"][2].set_ylim(0, 1)
    axes["row3"][2].set_ylabel("validation accuracy")

    runner_folders = [
        backprop,
        rnn_fast_mode_4_2,
        rflo_murray,
    ]
    labels = [
        "BPTT",
        "CS ($D_h=5$) w/ DFA",
        "RFLO",
    ]

    limit = 90
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
            # median = np.median(z, axis=0)
            all_values = np.append(all_values, average)
        all_values = all_values[:limit]
        x_axis = all_files_int[: len(all_values)]
        axes["row3"][3].plot(
            x_axis, all_values, label=labels[index], color=colors[index]
        )
        # plt.fill_between(x_axis, 0, all_values, alpha=0.5, facecolor=colors[index])
        axes["row3"][3].scatter(x_axis, all_values, color=colors[index], s=10)
        x_axis = np.array(x_axis)

    # axes["row3"][2].set_title(r"7 Timestep EMNIST Performance")
    axes["row3"][3].set_xlabel("training samples per class")
    # plt.axvline(x=5, color='blue', linestyle='--', label="Min Training Images for Small Examples")
    axes["row3"][3].axvline(
        x=30,
        color="black",
        linestyle="--",  # label="Min / Max Training Images"
    )
    axes["row3"][3].axvline(x=50, color="black", linestyle="--")
    axes["row3"][3].legend()
    axes["row3"][3].set_xlim(-5, limit + 10)
    axes["row3"][3].set_ylim(0, 1)
    axes["row3"][3].set_ylabel("validation accuracy")

    plt.tight_layout()
    plt.savefig(save_dir + "/all_7.png", dpi=300)
    plt.savefig(save_dir + "/all_7.pdf", dpi=300)
    plt.savefig(save_dir + "/all_7_2.svg", dpi=300)
    plt.show()

# Results for Computation in Biological NNs

This repository contains the results for the project "Computation in Biological Neural Networks".
Code for the project can be found [here](https://github.com/kmcm0707/Computation-in-Biological-Neural-Networks)

## Notes

All results so far are for the EMNIST dataset.

### Mode 1:

Equation 1: h(s+1) = yh(s) + (1/\eta) * zf(Kh(s) + \eta * P * F(Parameter) + b)
Equation 2: w(s) = v * h(s)

#### Results:

##### Baselines:
Mode_1_2: Baselines for Mode 1 with 1 - 6 Chemicals
Mode_1/Add: Mode 1 with 5 Chemicals
Mode_1/Baselines: Baselines for Mode 1 with 1 - 6 Chemicals

##### Initalisation Comparisons:

Mode_1/K_0.01 - K randomly initialised with 0.01 std - No Improvement
Mode_1/K_0_01_std_rosenbaum - K randomly initialised with 0.01 std, P initalised with rosenbaum end results - No Improvement
Mode_1/K_0.5 - K randomly initialised with 0.5 std - Maybe slight Improvement?
Mode_1/rosenbaum_first - P initalised with rosenbaum end results - Worse
Mode_1/random_P_0.01 - P randomly initialised with 0.01 std - Worse
Mode_1/random_P_0.01_Comment - P randomly initialised with 0.01 std but first coloumn posotive - Reaches convergence quicker but ends up worse
Mode_1/zeros_h - h initialised with zeros - No Improvement

kaiming - h initialised with kaiming - No Improvement or worse

##### Learning Rule Comparisons:

Mode_1/signal_oja_test - Signal Oja - sigmoid(y_[l+1])^Tsigmoid(y_l) - sigmoid(y_[l+1])^T W sigmoid(y_l) - Doesnt work
Mode_1/signal_hebb_1 - Signal Hebb - sigmoid(y_[l+1])^Tsigmoid(y_l) - 1 Chemical Signal - Worse
Mode_1/signal_hebb_2 - Signal Hebb - sigmoid(y_[l+1])^Tsigmoid(y_l) - 2 Chemical Signal - No Improvement

Mode_1/signal_hebb_1 and Mode_1/signal_hebb_2 also have most activation coloumn = (w-x) - this I remember worked a little better.

##### Bias:

Mode_1/update_bias - Neuron specific bias - 3 Chemicals - Huge improvement
Mode_1/mode_1_bias - Neuron specific bias - 5 Chemicals - Huge Improvement

##### Normalisation:

add_normalisation/error_and_activation - Normalisation of error and activation - No Improvement
add_normalisation/normilised_update - Normalisation of update - Slight Improvement?

add_normalisation/ bunch of others - Normalisations which didnt work
add_normalisation/data_normalise_test - turns out it was already normalised this just makes it much worse

##### Different Non-linearities:

Default non-linearity is tanh

Sigmoid - Instant NaN
ELU - Bit worse
GELU - Bit worse
ReLU - Doesn't work
Leaky ReLU - Doesn't work

##### Etas:

All in complex_eta for eta = 0.01 and 0.001

No Improvement

##### Reversed y and z:

Swap y and z in the update rule

In reversed_y_z - All a little worse

### Mode 2:
Equation 1: h(s+1) = yh(s) + (1/\eta) * zf(Kzh(s) + \eta * P * F(Parameter) + b)
Equation 2: w(s) = v * h(s)

Note mode 2 generalises to mode 1 basically.

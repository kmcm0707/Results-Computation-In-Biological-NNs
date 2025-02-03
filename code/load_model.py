import os

import torch
import torch.nn as nn
from complex_synapse import ComplexSynapse
from torch.utils.data import DataLoader, RandomSampler
from complex_options import complexOptions, nonLinearEnum, pMatrixEnum, kMatrixEnum, zVectorEnum, yVectorEnum, operatorEnum, vVectorEnum, modeEnum
from chemical_nn import ChemicalNN

if __name__ == "__main__":
    numberOfChemicals = 2
    modelOptions = complexOptions(
            nonLinear=nonLinearEnum.tanh,
            bias=False,
            update_rules=[0, 1, 2, 3, 4, 5, 6, 8, 9],
            pMatrix=pMatrixEnum.first_col,
            kMatrix=kMatrixEnum.zero,
            minTau=1,
            maxTau=50,
            y_vector=yVectorEnum.first_one,
            z_vector=zVectorEnum.default,
            operator=operatorEnum.mode_1,
            train_z_vector=False,
            mode=modeEnum.all,
            v_vector=vVectorEnum.default,
            eta=1,
    )
    model = ChemicalNN(device="cpu", numberOfChemicals=numberOfChemicals, small=False)
    complex = ComplexSynapse(
        device="cpu",
        numberOfChemicals=numberOfChemicals,
        complexOptions=modelOptions,
        params=model.named_parameters(),
    )
    directory = os.getcwd() + "/results/Mode_1/baselines/0/2"
    complex.load_state_dict(torch.load(directory + "/UpdateWeights.pth", map_location=torch.device("cpu"), weights_only=True))
    model.load_state_dict(torch.load(directory + "/model.pth", map_location=torch.device("cpu"), weights_only=True))
    print(torch.norm(model.chemical2[0], p=2))
    print(torch.norm(model.chemical2[1], p=2))

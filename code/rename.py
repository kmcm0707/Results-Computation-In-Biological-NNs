import os

if __name__ == '__main__':
    path = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results_runner\runner_min_tau_testing_60\0"
    all_files = os.listdir(path)
    rename = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190] # , 200, 250, 300, 350, 375
    print(len(all_files))
    print(len(rename))
    assert len(all_files) == len(rename)
    for i, j in zip(all_files, rename):
        os.rename(path + "/" + i, path + "/" + str(j))

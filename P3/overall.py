#!/usr/bin/python
import numpy as np

if __name__ == "__main__":

    direc = input("Stats Folder: ")
    f = np.load(f"{direc}/overall.npz")
    print("Overall Training Accuracy: ", np.around(np.sum(f['sub_train_acc'])/10.0, 3))
    print("Overall Test Accuracy: ", np.around(np.sum(f['sub_test_acc'])/10.0, 3))

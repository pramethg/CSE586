#!/usr/bin/python
import numpy as np

if __name__ == "__main__":

    direc = input("Stats Folder: ")
    data = input("Dataset: ")
    if data == "w":
        dataset = "Wallpaper"
    else:
        dataset = "Taiji"
    test_set = input("Provide the Test Set: ")
    f = np.load(f"{direc}/{dataset}/{test_set}/stats/overall.npz")
    if data == "w":
        print("Overall Training Accuracy: ", np.around(np.sum(f['classes_train'])/17.0, 5))
        print("Overall Test Accuracy: ", np.around(np.sum(f['classes_test'])/17.0, 5))
    else:
        print("Overall Training Accuracy: ", np.around(np.sum(f['sub_train_acc'])/10.0, 5))
        print("Overall Test Accuracy: ", np.around(np.sum(f['sub_test_acc'])/10.0, 5))

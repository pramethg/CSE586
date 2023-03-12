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
        print("Overall Training Accuracy: ", np.around(np.mean(f['classes_train']), 5))
        print("Overall Test Accuracy: ", np.around(np.mean(f['classes_test']), 5))
        print("Standard Deviation of Training Accuracy: ", np.around(np.std(f['classes_train']), 5))
        print("Standard Deviation of Test Accuracy: ", np.around(np.std(f['classes_test']), 5))
    else:
        print("Overall Training Accuracy: ", np.around(np.mean(f['sub_train_acc']), 5))
        print("Overall Test Accuracy: ", np.around(np.mean(f['sub_test_acc']), 5))
        print("Standard Deviation of Training Accuracy: ", np.around(np.std(f['sub_train_acc']), 5))
        print("Standard Deviation of Test Accuracy: ", np.around(np.std(f['sub_test_acc']), 5))

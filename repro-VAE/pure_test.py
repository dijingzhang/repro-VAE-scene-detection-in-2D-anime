import numpy as np
import os
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='11785_Project_test')

    parser.add_argument('--labels', type=str,
                        default='scene_change_label.npy')
    parser.add_argument('--directory', type=str,
                        default='/content/gdrive/MyDrive/11785-project/npy_folder/pure_tau')

opts = parser.parse_args()
labels = opts.labels
directory = opts.directory

total_correct = 0
for filename in sorted(os.listdir(directory))[12:22]:
    # model_path_list.append(os.path.join(directory, filename))
    results = np.load(os.path.join(directory, filename))

    correct = 0
    # results.sort()
    # print(results)
    # print(labels)
    for i in range(len(labels)):
        # if results[i][1] in labels:
        if results[i] in labels:
            correct += 1
    total_correct += correct
    print(filename, " Acc: ", correct / len(labels))


print("acc total:", total_correct / len(labels) / 10)
import numpy as np
import os
import csv
from torchvision import transforms as T
from PIL import Image
import torch.utils.data as data
import pandas as pd
import torch
import matplotlib.pyplot as plt


def save2csv(path, csvname='anime_data.csv'):
    # read imgname in dirt
    path_list = os.listdir(path)
    path_list.sort(key=lambda x:int(x.split('frame_')[1].split('.jpeg')[0]))
    # save img to csv
    f = open(csvname, 'a', newline='')
    for filename in path_list:
        writer = csv.writer(f)
        writer.writerow([filename])
    print("CSV file has been created!")
    f.close()
save2csv('frame/')   # only run this once! Need to comment it once you create the csv file!


# Image normalization and data augmentation
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
transform_init = T.Compose([T.ToTensor(), normalize, T.Resize((224, 224))])

# Define Dataset
# Maybe we can use ImageFolder and it will be faster than using Image to read imgs every time
class Dataload(data.Dataset):
    def __init__(self, csv_name, imgpath, transform=transform_init):
        super(Dataload, self).__init__()
        self.csv_name = csv_name
        self.transform = transform
        self.path = imgpath + np.array(pd.read_csv(csv_name, header=None))

    def __getitem__(self, index):

        img = Image.open(self.path[index][0])
        img = self.transform(img)
        return img.type('torch.FloatTensor')

    def __len__(self):
        return self.path.shape[0]

### set the range of frames you want to visualize
begin = 100
end = 120

dataset = Dataload('anime_data.csv', 'frame/')

L0 = []
L1 = []
L2 = []

for i in range(begin, end, 1):
    metric = torch.norm(dataset[i] - dataset[i + 1], 0)
    L0.append(metric.item())

    metric = torch.norm(dataset[i] - dataset[i + 1], 1)
    L1.append(metric.item())

    metric = torch.norm(dataset[i] - dataset[i + 1], 2)
    L2.append(metric.item())

min_L0, max_L0 = 0, 0
min_L1, max_L1 = 0, 0
min_L2, max_L2 = 0, 0

min_L0, max_L0 = min(min(L0), min_L0), max(max(L0), max_L0)
min_L1, max_L1 = min(min(L1), min_L1), max(max(L1), max_L1)
min_L2, max_L2 = min(min(L2), min_L2), max(max(L2), max_L2)

normalize = lambda X, mn, mx: [(x - mn)/(mx - mn) for x in X]

L0 = normalize(L0, min_L0, max_L0)
L1 = normalize(L1, min_L1, max_L1)
L2 = normalize(L2, min_L2, max_L2)

x = np.arange(begin, end, 1)
plt.rcParams["figure.figsize"] = (20,3)

plt.plot(x, L0, 'b')
plt.plot(x, L1, 'r')
plt.plot(x, L2, 'g')
plt.legend(['L0', 'L1', 'L2'])
plt.xticks(range(begin, end))
plt.show()

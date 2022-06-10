import torch
from lib.dataset.mixed_dataset import SingleDataset

dataset = SingleDataset("pw3d")

for data in dataset:
    for key in data.keys():
        print(key,data[key].shape)

    exit()
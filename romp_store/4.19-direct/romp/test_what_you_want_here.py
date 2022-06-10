import torch
from lib.dataset.mixed_dataset import SingleDataset
from torch.utils.data import Dataset, DataLoader, ConcatDataset

dataset = SingleDataset("pw3d",train_flag=False, mode='vibe', split='val')
loader = DataLoader(dataset = dataset, shuffle=True,batch_size = 1,\
                drop_last = False, pin_memory = True, num_workers = 2)

for data in loader:
    for i in range(3):
        print(data['imgpath'][i])

    exit()
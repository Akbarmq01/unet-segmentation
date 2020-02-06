import os
import argparse
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class DataFolder(Dataset):
    # return data in image, label format
    def __init__(self, opt, phase, transform=None):
        self.rootdir = opt.rootdir
        self.transform = transform
        self.data_path = os.path.join(self.rootdir, 'data')
        self.label_path = os.path.join(self.rootdir, 'labels')
        self.listdir = os.listdir(self.data_root)
        if phase








class LoadData:
    """
        Creates batch-wise dataloader
    """
    def __init__(self, opt):
        self.opt = opt
        self.params = {
            'batch_size': opt.batch_size,
            'num_workers': opt.num_workers,
            'shuffle': opt.shuffle
        }

    def __call__(self):
        datagen = {x: DataFolder(
            self.opt, x, transform=transforms.ToTensor()) for x in ['train', 'val']}
        return {x: torch.utils.data.DataLoader(datagen[x], **self.params) for x in ['train', 'val']}

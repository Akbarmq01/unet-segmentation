"""
    Data Processing classes
"""
import os
import argparse
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset

# - Create DataLoader


class DataFolder(Dataset):
    """
        Returns a datagenerator to
        generate data in the format
        (data, label)
    """

    def __init__(self, opt, phase, transform=None):
        """
            Data Generator parameter initialization
        """
        self.rootdir = opt.rootdir
        self.transform = transform

        self.data_root = os.path.join(self.rootdir, 'data')
        self.label_root = os.path.join(self.rootdir, 'labels')

        self.listdir = os.listdir(self.data_root)
        if phase == 'train':
            self.listdir = self.listdir[:int(len(self.listdir)*opt.split)]

        elif phase == 'val':
            self.listdir = self.listdir[int(len(self.listdir)*opt.split):]

        opt.dataset_size[phase] = len(self.listdir)
        
    def __len__(self):
        """
            Returns length of dataset
        """
        return len(self.listdir)

    def __getitem__(self, idx):
        """
            Returns image and corresponding label
        """
        data = cv2.imread(os.path.join(self.data_root, self.listdir[idx]), 0)
        label = cv2.imread(os.path.join(self.label_root, self.listdir[idx]), 0)
        if self.transform:
            data = self.transform(data)
            label = self.transform(label)
        return data, label

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--rootdir', default='/home/akbar/data/unet_dataset', help='dataset root directory(where folders having images and labels are there)')
        self.parser.add_argument('--split', default=0.80, type=float, help='Train/val split ratio')
        self.parser.add_argument('--batch_size', default=4, type=int, help='train batch size')
        self.parser.add_argument('--num_workers', default=8, type=int, help='thread count for training')
        self.parser.add_argument('--shuffle', default=True, help='flag to shuffle the dataset')
        self.parser.add_argument('--num_epochs', default=10, type=int, help='number of epochs to train')
        self.parser.add_argument('--image_size', default=512, type=int, help='input image size' )
        self.parser.add_argument('--in_channels', default=1, type=int, help='input channels of image')
        self.parser.add_argument('--num_class', default=1, type=int, help='number of classes in the dataset')
        self.opt = self.parser.parse_args()
        self.opt.dataset_size = {'train':None, 'val':None}

    def __call__(self):
        return self.opt


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
            self.opt, x, transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
            ])) for x in ['train', 'val']}
        return {x: torch.utils.data.DataLoader(datagen[x], **self.params) for x in ['train', 'val']}

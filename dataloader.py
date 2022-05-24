import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from skimage import io
import json


class MyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.frame.iloc[idx, 0][-16:]
        img_path = os.path.join(self.root_dir, img_name)
        image = io.imread(img_path)
        labels = self.frame.iloc[idx, 2]
        labels = json.loads(labels)[0]
        x, y, w, h = labels['x'], labels['y'], labels['width'], labels['height']
        angle1 = labels['rotation'] / 180 * np.pi
        angle0 = np.arctan(h/w)
        r = np.sqrt(w * w + h * h)
        x = labels['x'] + 0.5 * r * np.cos(angle1 + angle0)
        y = labels['y'] + 0.5 * r * np.sin(angle1 + angle0)
        rotation = labels['rotation'] if labels['rotation'] < 180 else labels['rotation'] - 360
        labels = np.array([x, y, rotation])
        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).to(torch.float32),
                'labels': torch.from_numpy(labels).to(torch.float32)}
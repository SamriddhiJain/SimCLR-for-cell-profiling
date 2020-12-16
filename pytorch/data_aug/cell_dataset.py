import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils
from imutils import paths

class CellDataset(Dataset):
    """cropped cell image dataset."""

    def __init__(self, path, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = path
        self.train_images = list(paths.list_images(path))
        self.transform = transform
        print(f"Total images: {len(self.train_images)}")

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_string = torch.Tensor(io.imread(self.train_images[idx]))
        image = torch.split(image_string, int(image_string.shape[1]/3), dim=1)
        sample = torch.stack((image[0], image[1], image[2])).float()
        sample = transforms.ToPILImage()(sample)
        sample = sample.resize((256, 256), 2)

        if self.transform:
            sample = self.transform(sample)

        return sample

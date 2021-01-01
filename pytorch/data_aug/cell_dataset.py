import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils
from imutils import paths
from tqdm import tqdm


class CellDataset(Dataset):
    """cropped cell image dataset."""

    def __init__(self, path, root_dir, input_shape, preload, transform=None):
        """
        Args:
            path (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.input_shape = input_shape
        self.root_dir = root_dir
        dataframe = pd.read_csv(path)
        self.labels = dataframe['Target']
        self.train_images = dataframe['Image_Name']
        self.preload = preload

        print(f"Total images: {len(self.train_images)}")
        if self.preload:
            print("Loading images ...")
            self.images = []
            for im_name in tqdm(self.train_images):
                sample = self.load_image(self.root_dir+im_name)
                self.images.append(sample)

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.preload:
            return self.images[idx], self.labels[idx]
        else:
            return self.load_image(self.root_dir+self.train_images[idx]), self.labels[idx]

    def load_image(self, path):
        image_string = torch.Tensor(io.imread(path))
        image = torch.split(image_string, int(image_string.shape[1]/3), dim=1)
        sample = torch.stack((image[0], image[1], image[2])).float()
        sample = transforms.ToPILImage()(sample)
        sample = sample.resize((self.input_shape[0], self.input_shape[0]), 2)

        if self.transform:
            sample = self.transform(sample)

        return sample

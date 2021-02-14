import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils
from imutils import paths
from tqdm import tqdm
from joblib import Parallel, delayed


class CellDataset(Dataset):
    """cropped cell image dataset."""

    def __init__(self, path, root_dir, input_shape, preload, num_workers, transform=None):
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
        self.num_workers = num_workers

        print(f"Total images: {len(self.train_images)}")
        if self.preload:
            print("Loading images ...")
            # self.images = Parallel(n_jobs=self.num_workers, verbose=55)(
            #     delayed(self.load_image)(self.root_dir+im_name) for im_name in self.train_images
            # )
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
            sample = self.images[idx]
        else:
            sample = self.load_image(self.root_dir+self.train_images[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample, self.labels[idx]

    def load_image(self, path):
        image_string = torch.Tensor(io.imread(path))
        image = torch.split(image_string, int(image_string.shape[1]/3), dim=1)
        sample = torch.stack((image[0], image[1], image[2])).float()
        sample = transforms.ToPILImage()(sample)
        sample = sample.resize((self.input_shape[0], self.input_shape[0]), 2)

        return sample

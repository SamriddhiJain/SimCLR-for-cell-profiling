import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import torchvision.transforms as transforms
from torch import DoubleTensor
from torchvision import datasets
from data_aug.cell_dataset import CellDataset

np.random.seed(0)


class DataSetWrapper(object):

    def __init__(self, batch_size, path, root_dir, num_workers, valid_size, input_shape, sampler, preload, **args):
        self.path = path
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.input_shape = eval(input_shape)
        self.sampler = sampler
        self.preload = preload

    def get_data_loaders(self):
        composed = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.7469, 0.7403, 0.7307), (0.1548, 0.1594, 0.1706))])
        train_dataset = CellDataset(self.path, self.root_dir, self.input_shape, self.preload,
                                    transform=composed)

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        if self.sampler == "random":
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
        elif self.sampler == "weighted":
            labels_df = train_dataset.labels.to_frame()
            weights = 1. / (int(labels_df.nunique()) * labels_df.groupby('Target')['Target'].transform('count'))
            # target 32 is "DMSO_0.0"
            epoch_size = len(labels_df[labels_df['Target'] != 32]) * int(labels_df.nunique()) / (int(labels_df.nunique())-1)

            train_sampler = WeightedRandomSampler(weights=DoubleTensor(list(weights[train_idx])),
                                                  num_samples=int(np.floor((1-self.valid_size) * epoch_size)),
                                                  replacement=False)
            valid_sampler = WeightedRandomSampler(weights=DoubleTensor(list(weights[valid_idx])),
                                                  num_samples=int(np.floor(self.valid_size * epoch_size)),
                                                  replacement=False)
        else:
            raise Exception(f"Sampler {self.sampler} is not supported.")

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets
from data_aug.cell_dataset import CellDataset

np.random.seed(0)

class DataSetWrapper(object):

    def __init__(self, batch_size, path, root_dir, num_workers, valid_size, input_shape, **args):
        self.path = path
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.input_shape = eval(input_shape)

    def get_data_loaders(self):
        composed = transforms.Compose([transforms.ToTensor()])
        train_dataset = CellDataset(self.path, self.root_dir, self.input_shape,
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
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms
from data_aug.custom import GaussianBlur, RandomCenterCrop
from torchvision import datasets
from data_aug.cell_dataset import CellDataset
from data_aug.dataset_wrapper import DataSetWrapper
import albumentations as transforms
from albumentations.pytorch import ToTensorV2
from PIL import Image

np.random.seed(0)


class DataSetWrapperSimCLR(DataSetWrapper):

    def __init__(self, batch_size, path, root_dir, num_workers, valid_size, input_shape, s, sampler, preload):
        super().__init__(batch_size, path, root_dir, num_workers, valid_size, input_shape, sampler, preload)
        self.s = s

    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()

        train_dataset = CellDataset(self.path, self.root_dir, self.input_shape, self.preload, self.num_workers,
                                    transform=SimCLRDataTransform(data_augment))

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        im_w = self.input_shape[0]
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(im_w, im_w),
                                              # transforms.Compose([RandomCenterCrop(),
                                              #                    transforms.Resize(size=self.input_shape[0])]),
                                              transforms.HorizontalFlip(),
                                              transforms.ElasticTransform(),
                                              # transforms.RandomRotate90(),
                                              transforms.ColorJitter(p=0.8),
                                              # transforms.RandomGrayscale(p=0.2),
                                              # transforms.Compose([RandomCenterCrop(),
                                              #                     transforms.Resize(size=self.input_shape[0])]),
                                              # transforms.Compose([transforms.CenterCrop(self.input_shape[0]//2),
                                              #                    transforms.Resize(size=self.input_shape[0])]),
                                              transforms.GaussianBlur(blur_limit=[int(0.1 * im_w), int(0.1 * im_w)],
                                                                      sigma_limit=[0.1, 2]),
                                              transforms.Normalize((0.7469, 0.7403, 0.7307), (0.1548, 0.1594, 0.1706)),
                                              ToTensorV2()
                                              ])
        return data_transforms


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(image=np.array(sample))["image"]
        xj = self.transform(image=np.array(sample))["image"]
        return xi, xj

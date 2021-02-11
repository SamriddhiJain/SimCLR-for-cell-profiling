import cv2
import numpy as np
import random

np.random.seed(0)


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class RandomCenterCrop(object):

    def __init__(self, scale=(0.08, 1.0)):
        self.scale = scale

    def __call__(self, sample):
        w, h = sample.size
        new_half_size = int(np.ceil(random.uniform(self.scale[0], self.scale[1]) * min(h, w) / 2))

        c_h = h // 2
        c_w = w // 2

        cropped = sample.crop((c_w - new_half_size, c_h - new_half_size,
                               c_w + new_half_size, c_h + new_half_size))

        return cropped

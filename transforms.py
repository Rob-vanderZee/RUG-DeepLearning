# Code (mostly) from BALRAJ ASHWATH
# source: https://www.kaggle.com/code/balraj98/pointnet-for-3d-object-classification-ii-pytorch

import numpy as np
import torch
from torchvision import transforms

from PointSampler import PointSampler


def assert_shape(pointcloud):
    assert len(pointcloud.shape) == 2


#
class Normalize(object):
    def __call__(self, pointcloud):
        assert_shape(pointcloud)
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
        return norm_pointcloud


class ToTensor(object):
    def __call__(self, pointcloud):
        assert_shape(pointcloud)
        return torch.from_numpy(pointcloud)


def default_transforms():
    return transforms.Compose([
            PointSampler(1024),
            Normalize(),
            ToTensor()
        ])


# def train_transforms():
#     return transforms.Compose([
#             PointSampler(1024),
#             Normalize(),
#             # RandRotation_z(),  # Why only rotate around z-axis?
#             # RandomNoise(),     # Do we want this?
#             ToTensor()
#         ])

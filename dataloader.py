# Code (mostly) from BALRAJ ASHWATH
# source: https://www.kaggle.com/code/balraj98/pointnet-for-3d-object-classification-ii-pytorch

import os
from torch.utils.data import Dataset, DataLoader
from path import Path

import transforms


def read_off(file):
    off_header = file.readline().strip()
    if 'OFF' == off_header:
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    else:
        n_verts, n_faces, __ = tuple([int(s) for s in off_header[3:].split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


class PointCloudData(Dataset):
    def __init__(self, root_dir, folder, transform):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {'pcd_path': new_dir / file, 'category': category}
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = read_off(file)
        pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud,
                'category': self.classes[category]}


def train_dataloader(root_dir, batch_size, shuffle=True):
    train_ds = PointCloudData(root_dir, folder='train', transform=transforms.default_transforms())
    return DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=shuffle)


def test_dataloader(root_dir, batch_size, shuffle=False):
    test_ds = PointCloudData(root_dir, folder='test', transform=transforms.default_transforms())
    return DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=shuffle)

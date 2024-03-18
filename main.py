from path import Path
import torch

from dataloader import train_dataloader, test_dataloader
from model import PointNet
import training


path = Path("./Data/Original/ModelNet40")

train_loader = train_dataloader(path, batch_size=32)
valid_loader = test_dataloader(path, batch_size=64)

pointnet = PointNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pointnet.to(device)

optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.0008)

training.train(pointnet, train_loader, optimizer)

from path import Path

from dataloader import train_dataloader, test_dataloader


path = Path("./Data/Original/ModelNet40")

train_loader = train_dataloader(path, batch_size=32)
valid_loader = test_dataloader(path, batch_size=64)

# import torch
# from torch.utils.data import Dataset, DataLoader
# import numpy as np

# DATAPATH = "data/corruptmnist_v1/"

# class CorruptMNIST(Dataset):
#     def __init__(self, train=True):
#         train_images, train_target = [], []
#         for i in range(6):
#             train_images.append(np.load(DATAPATH + f"train_images_{i}.pt"))
#             train_target.append(np.load(DATAPATH + f"train_target_{i}.pt"))
#         train_images = np.concatenate(train_images)
#         train_target = np.concatenate(train_target)
        
#         test_images: torch.Tensor = np.load(DATAPATH + "test_images.pt")
#         test_target: torch.Tensor = np.load(DATAPATH + "test_target.pt")
        
#         train_images = train_images.unsqueeze(1).float()
#         test_images = test_images.unsqueeze(1).float()
#         train_target = train_target.long()
#         test_target = test_target.long()

#         train_set = list(zip(train_images, train_target))
#         test_set = list(zip(test_images, test_target))
        
#         return train_set, test_set
        
#     def __getitem__(self, idx):
#         image = self.images[idx]
#         label = self.labels[idx]
#         return image, label

# # def get_data():
# #     # Initialize datasets
# #     train_dataset = CorruptMNIST(train=True)
# #     test_dataset = CorruptMNIST(train=False)
    
# #     # Create dataloaders
# #     train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# #     test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
# #     return train_loader, test_loader


# # def show_examples(data_loader):
# #     import matplotlib.pyplot as plt
    
# #     # Get a batch of images
# #     images, labels = next(iter(data_loader))
    
# #     # Create a grid of 3x3 images
# #     fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    
# #     for i, ax in enumerate(axes.flat):
# #         if i < len(images):
# #             ax.imshow(images[i].squeeze(), cmap='gray')
# #             ax.set_title(f'Label: {labels[i].item()}')
# #             ax.axis('off')
    
# #     plt.tight_layout()
# #     plt.show()

# # if __name__ == '__main__':
# #     train_loader, _ = get_data()
# #     show_examples(train_loader)

# def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
#     """Plot images and their labels in a grid."""
#     row_col = int(len(images) ** 0.5)
#     fig = plt.figure(figsize=(10.0, 10.0))
#     grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
#     for ax, im, label in zip(grid, images, target):
#         ax.imshow(im.squeeze(), cmap="gray")
#         ax.set_title(f"Label: {label.item()}")
#         ax.axis("off")
#     plt.show()


# if __name__ == "__main__":
#     train_set, test_set = CorruptMNIST()
#     print(f"Size of training set: {len(train_set)}")
#     print(f"Size of test set: {len(test_set)}")
#     print(f"Shape of a training point {(train_set[0][0].shape, train_set[0][1].shape)}")
#     print(f"Shape of a test point {(test_set[0][0].shape, test_set[0][1].shape)}")
#     show_image_and_target(train_set.tensors[0][:25], train_set.tensors[1][:25])

from __future__ import annotations

import matplotlib.pyplot as plt  # only needed for plotting
import torch
from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting

DATA_PATH = "data/corruptmnist_v1"


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test dataloaders for corrupt MNIST."""
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{DATA_PATH}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{DATA_PATH}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"{DATA_PATH}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{DATA_PATH}/test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set


def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    for ax, im, label in zip(grid, images, target):
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    train_set, test_set = corrupt_mnist()
    print(f"Size of training set: {len(train_set)}")
    print(f"Size of test set: {len(test_set)}")
    print(f"Shape of a training point {(train_set[0][0].shape, train_set[0][1].shape)}")
    print(f"Shape of a test point {(test_set[0][0].shape, test_set[0][1].shape)}")
    show_image_and_target(train_set.tensors[0][:25], train_set.tensors[1][:25])
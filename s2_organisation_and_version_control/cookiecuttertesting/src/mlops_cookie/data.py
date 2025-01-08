from pathlib import Path
import torch
import typer
from torch.utils.data import Dataset, TensorDataset


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize the data."""
    return (images - images.mean()) / images.std()


class CorruptMNISTDataset(Dataset):
    """Custom dataset for the corrupted MNIST data."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

        # self.train_images = torch.load(self.data_path / "train_images.pt")
        # self.train_target = torch.load(self.data_path / "train_target.pt")
        # self.test_images = torch.load(self.data_path / "test_images.pt")
        # self.test_target = torch.load(self.data_path / "test_target.pt")

        train_images, train_target = [], []
        for i in range(6):
            train_images.append(torch.load(f"{self.data_path}/train_images_{i}.pt"))
            train_target.append(torch.load(f"{self.data_path}/train_target_{i}.pt"))
        train_images = torch.cat(train_images)
        train_target = torch.cat(train_target)

        test_images: torch.Tensor = torch.load(f"{self.data_path}/test_images.pt")
        test_target: torch.Tensor = torch.load(f"{self.data_path}/test_target.pt")

        train_images = train_images.unsqueeze(1).float()
        test_images = test_images.unsqueeze(1).float()
        train_target = train_target.long()
        test_target = test_target.long()

        # Now assign to the object attributes
        self.train_images = train_images
        self.test_images = test_images
        self.train_target = train_target
        self.test_target = test_target

        # Noramlize the data
        self.train_images = normalize(self.train_images)
        self.test_images = normalize(self.test_images)

        self.train_target = train_target
        self.test_target = test_target

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.train_images)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.train_images[index], self.train_target[index]

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        torch.save(self.train_images, output_folder / "train_images.pt")
        torch.save(self.train_target, output_folder / "train_target.pt")
        torch.save(self.test_images, output_folder / "test_images.pt")
        torch.save(self.test_target, output_folder / "test_target.pt")

    def get_datasets(self) -> tuple[TensorDataset, TensorDataset]:
        """Return train and test datasets as TensorDatasets."""
        train_set = TensorDataset(self.train_images, self.train_target)
        test_set = TensorDataset(self.test_images, self.test_target)
        return train_set, test_set


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = CorruptMNISTDataset(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)

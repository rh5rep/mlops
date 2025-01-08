import torch
import typer
from model import Model

from data import CorruptMNISTDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    dataset = CorruptMNISTDataset(
        raw_data_path="/Users/rami/Documents/DTU/Jan25/dtu_mlops/s2_organisation_and_version_control/cookiecuttertesting/data/raw/corruptmnist_v1"
    )

    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = dataset.get_datasets()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    typer.run(evaluate)

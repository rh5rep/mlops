import torch
import typer
from data import corrupt_mnist
from model import Model
from torch import nn, optim
from torch.utils.data import DataLoader

app = typer.Typer()


@app.command()
def train(lr: float = 1e-4, batch_size: int = 32, epochs: int = 10, model_path: str = "model.pth"):
    """Train the model with given parameters."""
    # Get data
    train_set, _ = corrupt_mnist()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}: batch {batch_idx}, Loss: {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


@app.command()
def evaluate(model_path: str, batch_size: int = 32):
    """Evaluate the model from given checkpoint."""
    # Load test data
    _, test_set = corrupt_mnist()
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Load model
    model = Model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")


if __name__ == "__main__":
    app()

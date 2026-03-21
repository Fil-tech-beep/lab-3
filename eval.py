import torch
from torch import nn

from dataset.dataset_and_dataloader import build_dataloaders
from models.NN_class import CustomNet
from utils.path import DATA_DIR, DEFAULT_CHECKPOINT_PATH


def evaluate(model, val_loader, loss_fn, device):
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)


            predictions = model(inputs)
            loss = loss_fn(predictions, targets)


            val_loss += loss.item()
            predicted = predictions.argmax(dim=1)
            batch_size = targets.shape[0]
            num_correct = (predicted == targets).sum().item()

            total += batch_size
            correct += num_correct

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100.0 * correct / total

    print(f"Evaluation | Loss: {val_loss:.6f} | Acc: {val_accuracy:.2f}%")
    return val_loss, val_accuracy


if __name__ == "__main__":
    
    # load validation dataset
    train_loader, val_loader = build_dataloaders(
        data_root=DATA_DIR,
        batch_size=32,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build the empty model
    model = CustomNet().to(device)

    # Load the trained parameters into the empty model
    model.load_state_dict(torch.load(DEFAULT_CHECKPOINT_PATH, map_location=device))

    loss_fn = nn.CrossEntropyLoss()

    evaluate(model, val_loader, loss_fn, device)
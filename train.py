import os
import torch
from torch import nn
import wandb

from dataset.dataset_and_dataloader import build_dataloaders
from models.NN_class import CustomNet
from utils.path import DATA_DIR, DEFAULT_CHECKPOINT_PATH


def train(num_epochs, model, train_loader, loss_fn, optimizer, device, save_path):
    best_train_acc = 0.0

    # ensure checkpoints folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            # Device handling
            # I have a mac --> no cuda, but if you run on colab: use cuda
            inputs = inputs.to(device)
            targets = targets.to(device)

            # ------------------------------
            # Training heart
            # ------------------------------
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()



            running_loss += loss.item()
            predicted = predictions.argmax(dim=1)
            
            batch_size = targets.shape[0]
            num_correct = (predicted == targets).sum().item()

            total += batch_size
            correct += num_correct

        # finished all batches of 1 epoch
            # train loss = sum of loss of all batches in 1 epoch / number of batches in one epoch = avg loss each batch over 1 epoch
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total

        # keep wandb updated on the model scores so that it can keep track of training for us 
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy
            })

        print(f"Train Epoch: {epoch}/{num_epochs} | Avg loss one epoch: {train_loss:.6f} | Acc: {train_accuracy:.2f}%")


        # IF model training accuracy > best training accuracy --> save best TRAIN model in checkpoints
            # kinda doesn't make sense, it always gets better so it's just like
            # saving the most recent model but whatever
                # would be better after the epoch to validate the model (eval.py)
                    # (so with torch.no_grad() make predictions and compute loss)
                # on the validation dataset and keep the model with best validation score
        if train_accuracy > best_train_acc:
            best_train_acc = train_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model → {save_path}")

    print(f"\nBest training accuracy: {best_train_acc:.2f}%")
    return model


if __name__ == "__main__":
    
    # define all the bullshit you need in the training
        # dataset loading delegated to data/
    train_loader, val_loader = build_dataloaders(
        data_root=DATA_DIR,
        batch_size=32,
        num_workers=0
    )

        # device: if CPU or GPU --> you run it on colab so try to use GPU (cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # model defined in models/
    model = CustomNet().to(device)

        # loss function
    loss_fn = nn.CrossEntropyLoss()

        # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # number of epochs
    num_epochs = 10

    # needed to do the cool plots + track model training with wandb
    wandb.init(
        project="lab-3-cnn",
        config={
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "momentum": 0.9,
            "optimizer": "SGD",
            "architecture": "CustomNet",
            "dataset": "TinyImageNet"
        }
    )

    # wandb inspect the model --> track gradients/parameters. NOTE: THIS IS PRETTY HEAVY, COULD BE REMOVED
    wandb.watch(model, loss_fn, log="all", log_freq=100)


    train(
        num_epochs=num_epochs,
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        save_path=DEFAULT_CHECKPOINT_PATH
    )

    # tell wandb to fuck off
    wandb.finish()
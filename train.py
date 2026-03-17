from models.NN_class import CustomNet

import torch
from torch import nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

def train_dataset_and_dataloader():
    '''
    Build training dataset + dataloader
    '''

    transform = T.Compose([
        T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # root/{classX}/x001.jpg
    tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform)
    train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=32, shuffle=True, num_workers=0)

    
    return train_loader




def train(num_epochs, model, train_loader, loss_fn, optimizer, device, save_path):
    best_train_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            # I have a mac --> no cuda
            inputs = inputs.to(device)
            targets = targets.to(device)


            # training heart
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()


            running_loss += loss.item()
            predicted = predictions.argmax(dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # OR correct += (predicted == targets).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total
        best_train_acc = max(best_train_acc, train_accuracy)

        print(f"Train Epoch: {epoch}/{num_epochs} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%")


        # save ONLY if better --> Every time the model improves, it overwrites the same file
        if train_accuracy > best_train_acc:
            best_train_acc = train_accuracy
            torch.save(model.state_dict(), save_path)
            print("Saved new best model")


    print(f"\nBest training accuracy: {best_train_acc:.2f}%")
    
    return model




if __name__ == '__main__':
    train_loader = train_dataset_and_dataloader()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 10
    save_path = "/Users/filippomontecchi/Desktop/FAIMDL/LAB/L3/lab-3/checkpoints/model.pth"
    trained_model = train(num_epochs, model, train_loader, loss_fn, optimizer, device, save_path)
    
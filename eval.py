import torch

# Validation loop
def validate(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx,(inputs, targets) in enumerate(val_loader):
          # I have a mac --> no cuda
          inputs = inputs.to(device)
          targets = targets.to(device)


          # no gradients, only predict and look how much the trained model was wrong
          predictions = model(inputs)
          loss = loss_fn(predictions, targets)


          val_loss += loss.item()
          predicted = predictions.argmax(dim=1)
          total += targets.size(0)
          correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy


if __name__ == '__main__':
   
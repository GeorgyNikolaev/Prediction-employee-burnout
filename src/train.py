import torch
import torcheval.metrics.functional as metrics
from burnout_classifier import BurnoutClassifier


def train(model, train_loader, optimizer, criterion, device, epoch) -> BurnoutClassifier:
    model.train()
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print('Train Epoch: {}\tLoss: {:.6f}'.format(
            epoch,
            loss.item()))
    return model


def validate(model, val_loader, criterion, device) -> float:
    model.eval()
    threshold = 0.5
    inps = []
    outputs = []
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            inps.append(data)
            target = target.to(device)
            output = model(data)
            outputs.append(output)
    outs = torch.tensor([pred > threshold for preds in outputs for pred in preds]).to(device)
    best_auc = metrics.auc(inps, outs)
    for thrshold in [0.05 * i for i in range(1, 20)]:
        outs = torch.tensor([pred > thrshold for preds in outputs for pred in preds]).to(device)
        auc = metrics.auc(inps, outs)
        if auc > best_auc:
            best_auc = auc
            threshold = thrshold
    return threshold


def test(model, test_loader, criterion, device, threshold) -> None:
    model.eval()
    test_loss = 0
    correct = 0
    false_positive = 0
    false_negative = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = int(output > threshold)
            correct += pred == int(target.item())
            false_positive += pred == 1 and target.item() == 0
            false_negative += pred == 0 and target.item() == 1
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Precision: {}/{} ({:.0f}%), Recall: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset),
        correct / (correct + false_positive) * 100.,
        100. * correct / (correct + false_negative)))

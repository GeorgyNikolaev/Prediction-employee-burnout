import torch
import torcheval.metrics as metrics
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
    outputs = []
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            outputs.append(output)
    outs = torch.tensor([float(pred > threshold) for preds in outputs for pred in preds]).to(device)
    nn_outs = torch.tensor([pred for preds in outputs for pred in preds]).to(device)
    metric = metrics.BinaryAUROC()
    metric.update(nn_outs, outs)
    best_auc = metric.compute().item()
    for thrshold in [0.05 * i for i in range(1, 20)]:
        outs = torch.tensor([float(pred > thrshold) for preds in outputs for pred in preds]).to(device)
        metric.update(nn_outs, outs)
        auc = metric.compute().item()
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
    targets = []
    preds = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            targets.append(target)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = torch.tensor([float(out > threshold) for out in output]).to(device)
            preds.append(pred)
            correct += sum(pred[i] == int(target[i].item()) for i in range(len(pred)))
            false_positive += sum(pred[i] == 1 and target[i].item() == 0 for i in range(len(pred)))
            false_negative += sum(pred[i] == 0 and target[i].item() == 1 for i in range(len(pred)))
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%, Precision: {:.0f}%, Recall: {:.0f}%\n'.format(
        test_loss,
        100. * correct / len(test_loader.dataset),
        correct / (correct + false_positive) * 100.,
        100. * correct / (correct + false_negative)))
    targets = torch.cat(targets)
    preds = torch.cat(preds)
    print(targets, preds, sep='\n')

import torch
import torch.nn as nn
import torch.optim as optim
from train import train, validate, test
from dataset import HRAnalysisDataset
from burnout_classifier import BurnoutClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdma


def main():
    dataset = HRAnalysisDataset('dataset.csv')
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    train_loader, val_loader, test_loader = torch.random_split(data_loader, [0.7, 0.1, 0.2])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BurnoutClassifier(dataset[0][0].shape[0])
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    epochs = 32
    for epoch in tqdma(range(1, epochs + 1)):
        model = train(model, train_loader, optimizer, criterion, device, epoch)
        lr_scheduler.step(0.0)
    threshold = validate(model, val_loader, criterion, device)
    test(model, test_loader, criterion, device, threshold)
    model.save('burnout_model_state_dict.save')
    optimizer.save('burnout_optimizer_state_dict.save')


if __name__ == '__main__':
    main()

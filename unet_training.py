import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

from image_annotation_dataset import *


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    model = UNet(n_channels=1, n_classes=1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    DATASET_ROOT = "DatasetRoot"
    INPUT_IMAGES = f"{DATASET_ROOT}\\InputImages"
    OUTPUT_IMAGES = f"{DATASET_ROOT}\\OutputImages"
    train_dataset = ImageAnnotationDataset(INPUT_IMAGES, OUTPUT_IMAGES, ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    num_epochs = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    best_loss = np.inf
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_precision = 0
        epoch_recall = 0
        epoch_f1_score = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, masks)
            epoch_loss += loss.item()
            loss.backward()

            optimizer.step()

            preds = torch.sigmoid(outputs) > 0.5
            accuracy = (preds == masks).sum().item() / np.prod(masks.shape)
            epoch_accuracy += accuracy

            precision = precision_score(masks.cpu().view(-1) > 0.5, preds.cpu().view(-1) > 0.5)
            epoch_precision += precision

            recall = recall_score(masks.cpu().view(-1) > 0.5, preds.cpu().view(-1) > 0.5)
            epoch_recall += recall

            f1 = f1_score(masks.cpu().view(-1) > 0.5, preds.cpu().view(-1) > 0.5)
            epoch_f1_score += f1

        scheduler.step()

        epoch_loss /= len(train_loader)
        epoch_accuracy /= len(train_loader)
        epoch_precision /= len(train_loader)
        epoch_recall /= len(train_loader)
        epoch_f1_score /= len(train_loader)

        print(
            f'Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss} | Accuracy: {epoch_accuracy} | Precision: {epoch_precision} | Recall: {epoch_recall} | F1 Score: {epoch_f1_score}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'model_unet_best.pth')

    torch.save(model.state_dict(), 'model_unet_last.pth')

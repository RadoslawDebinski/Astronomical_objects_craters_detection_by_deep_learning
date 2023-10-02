import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision.transforms import ToTensor
from datetime import datetime
import numpy as np

from image_annotation_dataset import ImageAnnotationDataset
from CD_ResidualUNet.ResidualUNet import ResidualUNet


DATASET_ROOT = "DatasetRoot"
INPUT_IMAGES = f"{DATASET_ROOT}\\InputImages"
OUTPUT_IMAGES = f"{DATASET_ROOT}\\OutputImages"
MODEL_PATH = "Model"

NUM_EPOCHS = 3
N_CHANNELS = 1
N_CLASSES = 1
LEARNING_RATE = 0.0001
SCHEDULER_STEP_SIZE = 2
SCHEDULER_GAMMA = 0.1
BATCH_SIZE = 16
BIN_THRESHOLD = 0.5
SAVE_MODEL_ITER_NUM = 50


def train_ResidualUNet():
    model = ResidualUNet(n_channels=N_CHANNELS, n_classes=N_CLASSES)

    criterion = nn.BCEWithLogitsLoss()  # BCELoss?
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    train_dataset = ImageAnnotationDataset(INPUT_IMAGES, OUTPUT_IMAGES, ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    scaler = GradScaler()
    best_loss = np.inf

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    num_epochs = NUM_EPOCHS
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_precision = 0
        epoch_recall = 0
        epoch_f1_score = 0

        for idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if idx % SAVE_MODEL_ITER_NUM == 0:
                torch.save(model.state_dict(), f'{MODEL_PATH}/model_epoch_{epoch}.pth')
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(f"{MODEL_PATH}/train_info.txt", "a") as f:
                    f.write(f"{current_time} | Epoch: {epoch} | Iteration: {idx}\n")

                preds = torch.sigmoid(outputs) > BIN_THRESHOLD
                accuracy = (preds == masks).sum().item() / np.prod(masks.shape)
                epoch_accuracy += accuracy

                precision = precision_score(masks.cpu().view(-1) > BIN_THRESHOLD, preds.cpu().view(-1) > BIN_THRESHOLD)
                epoch_precision += precision

                recall = recall_score(masks.cpu().view(-1) > BIN_THRESHOLD, preds.cpu().view(-1) > BIN_THRESHOLD)
                epoch_recall += recall

                f1 = f1_score(masks.cpu().view(-1) > BIN_THRESHOLD, preds.cpu().view(-1) > BIN_THRESHOLD)
                epoch_f1_score += f1

        scheduler.step()

        epoch_loss /= len(train_loader)
        epoch_accuracy /= len(train_loader)
        epoch_precision /= len(train_loader)
        epoch_recall /= len(train_loader)
        epoch_f1_score /= len(train_loader)

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(f"{MODEL_PATH}/train_info.txt", "a") as f:
            f.write(f'{current_time} | Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss} | Accuracy: {epoch_accuracy} | Precision: {epoch_precision} | Recall: {epoch_recall} | F1 Score: {epoch_f1_score}\n')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f'{MODEL_PATH}/model_net_best.pth')

    torch.save(model.state_dict(), f'{MODEL_PATH}/model_net_last.pth')


if __name__ == "__main__":
    train_ResidualUNet()

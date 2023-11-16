import torch
import torch.nn as nn
from datetime import datetime


class ModelTrainer:
    """
    Functionality for training ML model
    """

    def __init__(self, device, model, train_loader, valid_loader, criterion, optimizer, scheduler):
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def load_state(self, model_path):
        """
        Loads epoch, model, optimizer and scheduler state from file
        """

        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['epoch']

    def save_model(self, epoch, path):
        """
        Saves epoch, model, optimizer and scheduler state to file
        """

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)

    def init_weights(self):
        """
        Applies initial weights to module
        """

        self.model.apply(self.fill_weights)

    @staticmethod
    def fill_weights(module):
        """
        Initializes weights of module using Kaiming Uniform initialization
        """

        if type(module) == nn.Conv2d or type(module) == nn.Linear:
            torch.nn.init.kaiming_uniform_(module.weight)

    def train(self, epoch, start_time, batch_size, save_path=".", save_interval=50):
        """
        Epoch training process
        """

        self.model.train()

        train_loss = 0.0
        train_precision = 0.0
        train_recall = 0.0
        train_f1 = 0.0

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, masks)
            batch_loss = loss.item()
            loss.backward()
            self.optimizer.step()

            TP = (masks * outputs).sum().item()
            FP = ((1 - masks) * outputs).sum().item()
            FN = (masks * (1 - outputs)).sum().item()

            batch_precision = TP / (TP + FP) if TP + FP > 0 else 0.0
            batch_recall = TP / (TP + FN) if TP + FN > 0 else 0.0
            batch_f1 = 2 * (batch_precision * batch_recall) / (batch_precision + batch_recall) if batch_precision + batch_recall > 0 else 0.0

            train_loss += batch_loss
            train_precision += batch_precision
            train_recall += batch_recall
            train_f1 += batch_f1

            if batch_idx % save_interval == 0:
                self.save_model(epoch, f'{save_path}/model_{start_time}_epoch_{epoch}.pth')

                current_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
                with open(f"{save_path}/train_info_{start_time}.txt", "a") as f:
                    info = f"{current_time} | Phase: train | Epoch: {epoch} | Images: {batch_idx * batch_size} | L: {batch_loss:.4f} <{(train_loss / (batch_idx+1)):.4f}> | P: {batch_precision:.4f} <{(train_precision / (batch_idx+1)):.4f}> | R: {batch_recall:.4f} <{(train_recall / (batch_idx+1)):.4f}> | F1: {batch_f1:.4f} <{(train_f1 / (batch_idx+1)):.4f}>\n"
                    print(info)
                    f.write(info)

        train_loss /= len(self.train_loader)
        train_precision /= len(self.train_loader)
        train_recall /= len(self.train_loader)
        train_f1 /= len(self.train_loader)

        self.scheduler.step()

        current_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        with open(f"{save_path}/train_info_{start_time}.txt", "a") as f:
            info = f"{current_time} | Phase: train | [END] Epoch: {epoch} | L: <{train_loss:.4f}> | P: <{train_precision:.4f}> | R: <{train_recall:.4f}> | F1: <{train_f1:.4f}>\n"
            print(info)
            f.write(info)

    def validate(self, epoch, start_time, batch_size, save_path=".", save_interval=50):
        """
        Epoch validating process
        """

        self.model.eval()

        valid_loss = 0.0
        valid_precision = 0.0
        valid_recall = 0.0
        valid_f1 = 0.0

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(self.valid_loader):
                images, masks = images.to(self.device), masks.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, masks)
                batch_loss = loss.item()

                TP = (masks * outputs).sum().item()
                FP = ((1 - masks) * outputs).sum().item()
                FN = (masks * (1 - outputs)).sum().item()

                batch_precision = TP / (TP + FP) if TP + FP > 0 else 0.0
                batch_recall = TP / (TP + FN) if TP + FN > 0 else 0.0
                batch_f1 = 2 * (batch_precision * batch_recall) / (batch_precision + batch_recall) if batch_precision + batch_recall > 0 else 0.0

                valid_loss += batch_loss
                valid_precision += batch_precision
                valid_recall += batch_recall
                valid_f1 += batch_f1

                if batch_idx % save_interval == 0:
                    current_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
                    with open(f"{save_path}/train_info_{start_time}.txt", "a") as f:
                        info = f"{current_time} | Phase: valid | Epoch: {epoch} | Images: {batch_idx * batch_size} | L: {batch_loss:.4f} <{(valid_loss / (batch_idx+1)):.4f}> | P: {batch_precision:.4f} <{(valid_precision / (batch_idx+1)):.4f}> | R: {batch_recall:.4f} <{(valid_recall / (batch_idx+1)):.4f}> | F1: {batch_f1:.4f} <{(valid_f1 / (batch_idx+1)):.4f}>\n"
                        print(info)
                        f.write(info)

            valid_loss /= len(self.valid_loader)
            valid_precision /= len(self.valid_loader)
            valid_recall /= len(self.valid_loader)
            valid_f1 /= len(self.valid_loader)

            current_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            with open(f"{save_path}/train_info_{start_time}.txt", "a") as f:
                info = f"{current_time} | Phase: valid | [END] Epoch: {epoch} | L: <{valid_loss:.4f}> | P: <{valid_precision:.4f}> | R: <{valid_recall:.4f}> | F1: <{valid_f1:.4f}>\n"
                print(info)
                f.write(info)

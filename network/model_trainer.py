import torch
import torch.nn as nn
from datetime import datetime


class ModelTrainer:
    """
    Functionality for training ML model
    """

    def __init__(self, device, model):
        self.device = device
        self.model = model

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    def init_train_modules(self, train_loader, valid_loader, criterion, optimizer, scheduler=None):
        """
        Initialize train/valid loader, criterion, optimizer and optionally scheduler
        """

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def init_test_modules(self, test_loader, criterion):
        """
        Initialize test loader and criterion
        """

        self.test_loader = test_loader
        self.criterion = criterion

    def load_state(self, model_path):
        """
        Loads epoch, model, optimizer and scheduler state from file
        """

        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
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
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else {},
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

    @staticmethod
    def calculate_metrics(outputs, masks):
        """
        Calculate precision, recall and F1-score
        """

        preds, binary_masks = torch.round(outputs), torch.round(masks)
        TP = ((preds == 1) & (binary_masks == 1)).sum().item()
        FP = ((preds == 1) & (binary_masks == 0)).sum().item()
        FN = ((preds == 0) & (binary_masks == 1)).sum().item()

        # =========================================
        # Another method
        # TP = (masks * outputs).sum().item()
        # FP = ((1 - masks) * outputs).sum().item()
        # FN = (masks * (1 - outputs)).sum().item()
        # =========================================

        precision = TP / (TP + FP) if TP + FP > 0 else 0.0
        recall = TP / (TP + FN) if TP + FN > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

        return precision, recall, f1

    @staticmethod
    def save_batch_info(epoch, phase, logs_path,
                        batch_idx, batch_size,
                        batch_loss, batch_precision, batch_recall, batch_f1,
                        sum_loss, sum_precision, sum_recall, sum_f1):
        """
        Print and save to file batch statistics
        """

        current_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        with open(logs_path, "a") as f:
            info = (
                f"{current_time} | "
                f"Phase: {phase} | "
                f"Epoch: {epoch} | "
                f"Images: {batch_idx * batch_size} | "
                f"L: {batch_loss:.4f} <{(sum_loss / (batch_idx + 1)):.4f}> | "
                f"P: {batch_precision:.4f} <{(sum_precision / (batch_idx + 1)):.4f}> | "
                f"R: {batch_recall:.4f} <{(sum_recall / (batch_idx + 1)):.4f}> | "
                f"F1: {batch_f1:.4f} <{(sum_f1 / (batch_idx + 1)):.4f}>\n"
            )
            print(info)
            f.write(info)

    @staticmethod
    def save_epoch_info(epoch, phase, logs_path,
                        loss, precision, recall, f1):
        """
        Print and save to file epoch statistics
        """

        current_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        with open(logs_path, "a") as f:
            info = (
                f"{current_time} | "
                f"Phase: {phase} | "
                f"[END] "
                f"Epoch: {epoch} | "
                f"L: <{loss:.4f}> | "
                f"P: <{precision:.4f}> | "
                f"R: <{recall:.4f}> | "
                f"F1: <{f1:.4f}>\n"
            )
            print(info)
            f.write(info)

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

            batch_precision, batch_recall, batch_f1 = self.calculate_metrics(outputs, masks)

            train_loss += batch_loss
            train_precision += batch_precision
            train_recall += batch_recall
            train_f1 += batch_f1

            if batch_idx % save_interval == 0:
                self.save_model(epoch, f'{save_path}/model_{start_time}_temp.pth')
                self.save_batch_info(epoch, "train", f"{save_path}/train_info_{start_time}.txt",
                                     batch_idx, batch_size,
                                     batch_loss, batch_precision, batch_recall, batch_f1,
                                     train_loss, train_precision, train_recall, train_f1)

        train_loss /= len(self.train_loader)
        train_precision /= len(self.train_loader)
        train_recall /= len(self.train_loader)
        train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall)

        if self.scheduler is not None:
            self.scheduler.step()

        self.save_epoch_info(epoch, "train", f"{save_path}/train_info_{start_time}.txt",
                             train_loss, train_precision, train_recall, train_f1)

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

                batch_precision, batch_recall, batch_f1 = self.calculate_metrics(outputs, masks)
                valid_loss += batch_loss
                valid_precision += batch_precision
                valid_recall += batch_recall
                valid_f1 += batch_f1

                if batch_idx % save_interval == 0:
                    self.save_batch_info(epoch, "valid", f"{save_path}/train_info_{start_time}.txt",
                                         batch_idx, batch_size,
                                         batch_loss, batch_precision, batch_recall, batch_f1,
                                         valid_loss, valid_precision, valid_recall, valid_f1)

            valid_loss /= len(self.valid_loader)
            valid_precision /= len(self.valid_loader)
            valid_recall /= len(self.valid_loader)
            valid_f1 = 2 * valid_precision * valid_recall / (valid_precision + valid_recall)

            self.save_epoch_info(epoch, "valid", f"{save_path}/train_info_{start_time}.txt",
                                 valid_loss, valid_precision, valid_recall, valid_f1)

    def test(self, start_time, batch_size, save_path=".", save_interval=50):
        """
        Epoch testing process
        """

        self.model.eval()

        test_loss = 0.0
        test_precision = 0.0
        test_recall = 0.0
        test_f1 = 0.0

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(self.test_loader):
                images, masks = images.to(self.device), masks.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, masks)
                batch_loss = loss.item()

                batch_precision, batch_recall, batch_f1 = self.calculate_metrics(outputs, masks)
                test_loss += batch_loss
                test_precision += batch_precision
                test_recall += batch_recall
                test_f1 += batch_f1

                if batch_idx % save_interval == 0:
                    self.save_batch_info("-", "test", f"{save_path}/test_info_{start_time}.txt",
                                         batch_idx, batch_size,
                                         batch_loss, batch_precision, batch_recall, batch_f1,
                                         test_loss, test_precision, test_recall, test_f1)

            test_loss /= len(self.test_loader)
            test_precision /= len(self.test_loader)
            test_recall /= len(self.test_loader)
            test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall)

            self.save_epoch_info("-", "test", f"{save_path}/test_info_{start_time}.txt",
                                 test_loss, test_precision, test_recall, test_f1)

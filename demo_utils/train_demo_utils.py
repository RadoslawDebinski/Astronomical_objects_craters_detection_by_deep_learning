import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from matplotlib import pyplot as plt
from PIL import Image
from datetime import datetime

from dataset_creation.image_annotation_dataset import ImageAnnotationDataset
from network.attention_unet import AttentionUNet
from network.combo_loss import ComboLoss
from network.model_trainer import ModelTrainer
from settings import CONST_PATH, NET_PARAMS, OPTIM_PARAMS, SCHED_PARAMS, TRAIN_PARAMS


def train_model_const(prev_model_path=None):
    """
    Training demonstration with constants and params from 'settings'
    """

    train_in_path, train_out_path, save_model_path = (CONST_PATH["trainIN"], CONST_PATH["trainOUT"],
                                                      CONST_PATH["model"])
    num_epochs, batch_size, save_interval_iter = (TRAIN_PARAMS["num_epochs"], TRAIN_PARAMS["batch_size"],
                                                  TRAIN_PARAMS["save_interval_iter"])
    in_channels, out_channels, filters_num, dropout_p = (NET_PARAMS["in_channels"], NET_PARAMS["out_channels"],
                                                         NET_PARAMS["filters_num"], NET_PARAMS["dropout_p"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionUNet(in_channels, out_channels, filters_num, dropout_p)

    train_dataset = ImageAnnotationDataset(train_in_path, train_out_path)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_PARAMS["batch_size"], shuffle=True)

    valid_dataset = ImageAnnotationDataset(CONST_PATH["validIN"], CONST_PATH["validOUT"])
    valid_loader = DataLoader(valid_dataset, batch_size=TRAIN_PARAMS["batch_size"], shuffle=False)

    criterion = ComboLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=OPTIM_PARAMS["learning_rate"])

    # [Scheduler: last optional argument in init_train_modules() of ModelTrainer()]
    # sched = lr_scheduler.CosineAnnealingLR(optimizer, T_max=SCHED_PARAMS["t_max"], eta_min=SCHED_PARAMS["eta_min"])

    trainer = ModelTrainer(device, model)
    trainer.init_train_modules(train_loader, valid_loader, criterion, optimizer)
    trainer.init_weights()
    trainer.model = trainer.model.to(trainer.device)

    start_epoch = trainer.load_state(prev_model_path) if prev_model_path else 0
    start_time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')

    for epoch in range(start_epoch+1, num_epochs+1):
        trainer.train(epoch, start_time, batch_size, save_model_path, save_interval_iter)
        trainer.validate(epoch, start_time, batch_size, save_model_path, save_interval_iter)
        trainer.save_model(epoch, f'{save_model_path}/model_{start_time}_epoch_{epoch}_of_{num_epochs}.pth')

def test_model_const(model_path, astro_object="moon"):
    """
    Test trained model on data from astronomic object (Moon or Mars)
    """

    start_time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    if astro_object == "moon":
        test_in_path, test_out_path = CONST_PATH["testIN"], CONST_PATH["testOUT"]
    elif astro_object == "mars":
        test_in_path, test_out_path = CONST_PATH["marsIN"], CONST_PATH["marsOUT"]
    else:
        return

    batch_size, save_logs_path = TRAIN_PARAMS["batch_size"], CONST_PATH["model"]
    in_channels, out_channels, filters_num, dropout_p = (NET_PARAMS["in_channels"], NET_PARAMS["out_channels"],
                                                         NET_PARAMS["filters_num"], NET_PARAMS["dropout_p"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionUNet(in_channels, out_channels, filters_num, dropout_p)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model = model.to(device)
    model.eval()

    test_dataset = ImageAnnotationDataset(test_in_path, test_out_path)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_PARAMS["batch_size"], shuffle=True)

    criterion = ComboLoss()
    trainer = ModelTrainer(device, model)
    trainer.init_test_modules(test_loader, criterion)
    trainer.model = trainer.model.to(trainer.device)

    trainer.test(start_time, batch_size, save_logs_path)

def check_model_const(model_path, input_image, output_mask):
    """
    Checks trained model output mask and compares with expected mask
    """

    in_channels, out_channels, filters_num, dropout_p = (NET_PARAMS["in_channels"], NET_PARAMS["out_channels"],
                                                         NET_PARAMS["filters_num"], NET_PARAMS["dropout_p"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionUNet(in_channels, out_channels, filters_num, dropout_p)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model = model.to(device)
    model.eval()

    original = Image.open(input_image)
    expected = Image.open(output_mask)

    transform = Compose([ToTensor()])
    original = transform(original).unsqueeze(0).to(device)
    expected = transform(expected).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted = model(original)

    precision, recall, f1 = ModelTrainer.calculate_metrics(predicted, expected)
    print(f"P = {precision} | R = {recall} | F1 = {f1}")

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(original.data.cpu().numpy().squeeze(), cmap='gray')
    axs[0].title.set_text('Original')
    axs[1].imshow(expected.data.cpu().numpy().squeeze(), cmap='gray')
    axs[1].title.set_text('Expected')
    axs[2].imshow(predicted.data.cpu().numpy().squeeze(), cmap='gray')
    axs[2].title.set_text('Predicted')
    plt.show()

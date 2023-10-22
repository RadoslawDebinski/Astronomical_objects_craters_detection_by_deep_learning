import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from matplotlib import pyplot as plt
from PIL import Image
from datetime import datetime

from dataset_creation.image_annotation_dataset import ImageAnnotationDataset
from network.attention_unet import AttentionUNet
from network.bce_dice_loss import BCEDiceLoss
from network.model_trainer import ModelTrainer
from settings import CONST_PATH, NET_PARAMS, OPTIM_PARAMS, SCHED_PARAMS, TRAIN_PARAMS


def train_model_const(prev_model_path=None):
    """
    Training demonstration with constants and params from 'settings'
    """

    train_in_path, train_out_path, save_model_path = CONST_PATH["trainIN"], CONST_PATH["trainOUT"], CONST_PATH["model"]
    num_epochs, batch_size, save_interval_iter = TRAIN_PARAMS["num_epochs"], TRAIN_PARAMS["batch_size"], TRAIN_PARAMS["save_interval_iter"]
    in_channels, out_channels, filters_num = NET_PARAMS["in_channels"], NET_PARAMS["out_channels"], NET_PARAMS["filters_num"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionUNet(in_channels, out_channels, filters_num)

    train_dataset = ImageAnnotationDataset(train_in_path, train_out_path)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_PARAMS["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True, prefetch_factor=2)

    valid_dataset = ImageAnnotationDataset(CONST_PATH["validIN"], CONST_PATH["validOUT"])
    valid_loader = DataLoader(valid_dataset, batch_size=TRAIN_PARAMS["batch_size"], shuffle=False,
                              num_workers=2, pin_memory=True, prefetch_factor=2)

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=OPTIM_PARAMS["learning_rate"], weight_decay=OPTIM_PARAMS["weight_decay"])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=SCHED_PARAMS["t_max"], eta_min=SCHED_PARAMS["eta_min"])

    trainer = ModelTrainer(device, model, train_loader, valid_loader, criterion, optimizer, scheduler)
    trainer.init_weights()

    start_epoch = trainer.load_state(prev_model_path) if prev_model_path else 0
    start_time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')

    for epoch in range(start_epoch, num_epochs):
        trainer.train(epoch, start_time, batch_size, save_model_path, save_interval_iter)
        trainer.validate(epoch, start_time, batch_size, save_model_path, save_interval_iter)
        trainer.save_model(epoch, f'{save_model_path}/model_{start_time}_epoch_{epoch}_of_{num_epochs}.pth')


def check_model_const(model_path, input_image, output_mask):
    """
    Checks trained model output mask and compares with expected mask
    """

    in_channels, out_channels, filters_num = NET_PARAMS["in_channels"], NET_PARAMS["out_channels"], NET_PARAMS["filters_num"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionUNet(in_channels, out_channels, filters_num)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model = model.to(device)
    model.eval()

    image = Image.open(input_image)
    transform = Compose([ToTensor()])
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        mask_pred = torch.sigmoid(output).data.cpu().numpy()

    expected = Image.open(output_mask)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(image.cpu().squeeze(), cmap='gray')
    axs[0].title.set_text('Original')
    axs[1].imshow(expected, cmap='gray')
    axs[1].title.set_text('Expected')
    axs[2].imshow(mask_pred.squeeze(), cmap='gray')
    axs[2].title.set_text('Predicted')
    plt.show()

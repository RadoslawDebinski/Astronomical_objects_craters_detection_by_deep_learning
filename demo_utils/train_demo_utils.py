import os
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from datetime import datetime

from dataset_creation.image_annotation_dataset import ImageAnnotationDataset
from network.attention_unet import AttentionUNet
from network.combo_loss import ComboLoss
from network.model_trainer import ModelTrainer
from settings import CONST_PATH, NET_PARAMS, COMBO_LOSS_PARAMS, OPTIM_PARAMS, SCHED_PARAMS, TRAIN_PARAMS


def train_model_const(prev_model_path=None):
    """
    Training demonstration with constants and params from 'settings'
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionUNet(NET_PARAMS["in_channels"], NET_PARAMS["out_channels"],
                          NET_PARAMS["filters_num"], NET_PARAMS["p_drop"])

    train_dataset = ImageAnnotationDataset(CONST_PATH["trainIN"], CONST_PATH["trainOUT"])
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_PARAMS["batch_size"], shuffle=True)

    valid_dataset = ImageAnnotationDataset(CONST_PATH["validIN"], CONST_PATH["validOUT"])
    valid_loader = DataLoader(valid_dataset, batch_size=TRAIN_PARAMS["batch_size"], shuffle=False)

    criterion = ComboLoss(COMBO_LOSS_PARAMS["alpha"], COMBO_LOSS_PARAMS["beta"])
    optimizer = torch.optim.Adam(model.parameters(), lr=OPTIM_PARAMS["learning_rate"],
                                 weight_decay=OPTIM_PARAMS["weight_decay"])

    # [Scheduler: last optional argument in init_train_modules() of ModelTrainer()]
    # sched = lr_scheduler.CosineAnnealingLR(optimizer, T_max=SCHED_PARAMS["t_max"], eta_min=SCHED_PARAMS["eta_min"])

    trainer = ModelTrainer(device, model)
    trainer.init_train_modules(train_loader, valid_loader, criterion, optimizer)
    trainer.init_weights()
    trainer.model = trainer.model.to(trainer.device)

    start_epoch = trainer.load_state(prev_model_path) if prev_model_path else 0
    start_time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')

    save_model_path, save_interval_iter = CONST_PATH["model"], TRAIN_PARAMS["save_interval_iter"]
    num_epochs, batch_size  = TRAIN_PARAMS["num_epochs"], TRAIN_PARAMS["batch_size"]
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionUNet(NET_PARAMS["in_channels"], NET_PARAMS["out_channels"],
                          NET_PARAMS["filters_num"], NET_PARAMS["p_drop"])
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model = model.to(device)
    model.eval()

    test_dataset = ImageAnnotationDataset(test_in_path, test_out_path)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_PARAMS["batch_size"], shuffle=True)

    criterion = ComboLoss(COMBO_LOSS_PARAMS["alpha"], COMBO_LOSS_PARAMS["beta"])
    trainer = ModelTrainer(device, model)
    trainer.init_test_modules(test_loader, criterion)
    trainer.model = trainer.model.to(trainer.device)

    trainer.test(start_time, TRAIN_PARAMS["batch_size"], CONST_PATH["model"])

def check_model_const(model_path, input_image, output_mask, show=True, short_print=False):
    """
    Checks trained model output mask and compares with expected mask
    """

    start_time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    results_path = f"{CONST_PATH['model']}/results_{start_time}"
    os.makedirs(results_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_filename = os.path.splitext(os.path.basename(input_image))[0]
    original = Image.open(input_image)
    expected = Image.open(output_mask)

    transform = Compose([ToTensor()])
    original = transform(original).unsqueeze(0).to(device)
    expected = transform(expected).unsqueeze(0).to(device)

    checkpoint = torch.load(model_path)
    if 'model' in checkpoint:
        model = checkpoint['model']
        predicted = model(original)
    elif 'model_state_dict' in checkpoint:
        model = AttentionUNet(NET_PARAMS["in_channels"], NET_PARAMS["out_channels"],
                              NET_PARAMS["filters_num"], NET_PARAMS["p_drop"])
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            predicted = model(original)
    else:
        print("Something wrong with model! Try another...")
        return

    precision, recall, f1 = ModelTrainer.calculate_metrics(predicted, expected)
    precision = round(precision * 100, 2)
    recall = round(recall * 100, 2)
    f1 = round(f1 * 100, 2)
    print(f"P = {precision}% | R = {recall}% | F1 = {f1}%")

    # Masking

    original_data = original.data.cpu().numpy().squeeze()
    expected_data = expected.data.cpu().numpy().squeeze()
    predicted_data = predicted.data.cpu().numpy().squeeze()
    predicted_data_save = Image.fromarray((predicted_data * 255).astype('uint8'))
    predicted_data_save.save(f"{results_path}/results-mask-{input_filename}.jpg")

    expected_mask = cv2.normalize(expected_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    predicted_mask = cv2.normalize(predicted_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    threshold1, threshold2, color1, color2, color3 = 127, 127, [0, 0, 255], [255, 0, 0], [0, 255, 0]
    expected_mask_colored = cv2.cvtColor(expected_mask, cv2.COLOR_GRAY2BGR)
    expected_mask_colored[expected_mask > threshold1] = np.array(color1)
    predicted_mask_colored = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2BGR)
    predicted_mask_colored[predicted_mask > threshold1] = np.array(color2)

    final_mask = cv2.add(expected_mask_colored, predicted_mask_colored)
    red_in_expected_mask = np.all(expected_mask_colored == color1, axis=-1)
    blue_in_predicted_mask = np.all(predicted_mask_colored == color2, axis=-1)
    final_mask[np.logical_and(red_in_expected_mask, blue_in_predicted_mask)] = color3

    original_data_masked = cv2.normalize(original_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    original_data_masked = cv2.cvtColor(original_data_masked, cv2.COLOR_GRAY2BGR)
    original_data_masked[final_mask > threshold2] = final_mask[final_mask > threshold2]

    # Plotting

    titles = ['a)', 'b)', 'c)', 'd)'] if short_print else ['Original', 'Expected', 'Predicted', 'Comparing']
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    axs[0].imshow(original_data, cmap='gray')
    axs[0].title.set_text(titles[0])
    axs[1].imshow(expected_data, cmap='gray')
    axs[1].title.set_text(titles[1])
    axs[2].imshow(predicted_data, cmap='gray')
    axs[2].title.set_text(titles[2])
    axs[3].imshow(original_data_masked)
    axs[3].title.set_text(titles[3])

    plt.subplots_adjust(wspace=0.5)
    for ax in axs:
        ax.axis('off')
    fig.tight_layout()

    results_filename = (f"results-"
                        f"F1-{str(f1).replace('.', '_')}-"
                        f"P-{str(precision).replace('.', '_')}-"
                        f"R-{str(recall).replace('.', '_')}-"
                        f"{input_filename}.png")
    fig.savefig(f"{results_path}/{results_filename}", dpi=300)

    if show:
        plt.show()

    plt.close(fig)

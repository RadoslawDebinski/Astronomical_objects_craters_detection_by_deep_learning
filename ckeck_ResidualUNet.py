import torch
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Compose
import matplotlib.pyplot as plt

from CD_ResidualUNet.ResidualUNet import ResidualUNet


def check_ResidualUNet(model_name, input_image, expected_output_mask):
    """One example only"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    model = ResidualUNet(n_channels=1, n_classes=1)
    model.load_state_dict(torch.load(model_name))
    model = model.to(device)
    model.eval()

    image = Image.open(input_image).convert('L')
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])  # Normalizing to [-1, 1]
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        mask_pred = torch.sigmoid(output).data.cpu().numpy()

    expected = Image.open(expected_output_mask).convert('L')

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(image.cpu().squeeze(), cmap='gray')
    axs[0].title.set_text('Original')
    axs[1].imshow(expected, cmap='gray')
    axs[1].title.set_text('Expected')
    axs[2].imshow(mask_pred.squeeze(), cmap='gray')
    axs[2].title.set_text('Predicted')
    plt.show()


if __name__ == "__main__":
    check_ResidualUNet('Model/model_epoch_0.pth',
                       './DatasetRoot/InputImages/1_0000.jpg',
                       './DatasetRoot/OutputImages/1_0000.jpg')

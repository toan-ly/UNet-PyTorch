import torch
import time
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from load_dataset import CarvanaDataset
from unet import UNet

def pred_show_image_grid(data_path, model_pth, device):
    """
    Predict and show a grid of original images, original masks, and predicted masks.
    """
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    image_dataset = CarvanaDataset(data_path, test=True)
    
    images = []
    orig_masks = []
    pred_masks = []
    
    for img, orig_mask in image_dataset:
        img = img.float().to(device).unsqueeze(0)
        pred_mask = model(img)

        img = img.squeeze(0).cpu().detach().permute(1, 2, 0)

        pred_mask = pred_mask.squeeze(0).cpu().detach().permute(1, 2, 0)
        pred_mask = (pred_mask > 0).float()

        orig_mask = orig_mask.cpu().detach().permute(1, 2, 0)

        images.append(img)
        orig_masks.append(orig_mask)
        pred_masks.append(pred_mask)

    images.extend(orig_masks)
    images.extend(pred_masks)

    fig = plt.figure(figsize=(15, 5))
    total_images = len(image_dataset)
    for i in range(1, 3 * total_images + 1):
        ax = fig.add_subplot(3, total_images, i)
        ax = axis('off')
        plt.imshow(images[i-1], cmap='gray')
    plt.tight_layout()
    plt.show()

def single_image_inference(image_pth, model_pth, device):
    """
    Perform inference on a single image and display the original image and predicted mask.
    """
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])    

    img = transform(Image.open(image_pth)).float().to(device).unsqueeze(0)
    pred_mask = model(img)    

    img = img.squeeze(0).cpu().detach().permute(1, 2, 0)
    pred_mask = pred_mask.squeeze(0).cpu().detach().permute(1, 2, 0)
    pred_mask = (pred_mask > 0).float()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Original Image')
    ax[1].imshow(pred_mask, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('Predicted Mask')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    start_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time))
    print('-------------------------------------')
    print('Running Inference script at', start_time_str, '\n')
    
    SINGLE_IMG_PATH = './data/manual_test/...'
    DATA_PATH = './data'
    MODEL_PATH = './models/unet.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pred_show_image_grid(DATA_PATH, MODEL_PATH, device)
    single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device)

    end_time = time.time()
    end_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(end_time))
    print('\nDone with Inference at', end_time_str, '!')




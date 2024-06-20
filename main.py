import torch
import time
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from load_dataset import CarvanaDataset

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs):
    '''
    Train the model.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (optim.Optimizer): Optimizer to use for training.
        criterion (nn.Module): Loss function.
        device (str): Device to run the training on.
        epochs (int): Number of epochs to train.

    Returns:
        nn.Module: Trained model.
    '''
    for epoch in range(epochs):
        model.train()
        train_running_loss = 0.0
        for img_mask in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{epochs}'):
            imgs, masks = img_mask
            imgs, masks = imgs.float().to(device), masks.float().to(device)

            optimizer.zero_grad()

            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
        train_loss = train_running_loss / len(train_loader)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for img_mask in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{epochs}'):
                imgs, masks = img_mask
                imgs, masks = imgs.float().to(device), masks.float().to(device)

                preds = model(imgs)
                loss = criterion(preds, masks)

                val_running_loss += loss.item()     
        val_loss = val_running_loss / len(val_loader)
        
        print('-' * 30)
        print(f'Train Loss Epoch {epoch+1}: {train_loss:.4f}')
        print(f'Val Loss Epoch {epoch+1}: {val_loss:.4f}')
        print('-' * 30)

    return model


if __name__ == '__main__':
    start_time = time.time()
    start_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time))
    print('-------------------------------------')
    print('Running Training script at', start_time_str, '\n')
    
    # Hyperparameters and configuration
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 32
    EPOCHS = 2
    DATA_PATH = '/content/drive/MyDrive/unet-segmentation/data'
    MODEL_SAVE_PATH = '/content/drive/MyDrive/unet-segmentation/models/unet.pth'

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset and DataLoader setup
    dataset = CarvanaDataset(DATA_PATH)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        dataset, [0.8, 0.2], generator=generator)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False)

    # Model, loss, and optimizer setup
    model = UNet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Train the model
    model = train_model(model,
                        train_loader,
                        val_loader,
                        optimizer=optimizer,
                        criterion=criterion,
                        device=device,
                        epochs=EPOCHS)
    
    # Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'Model saved to {MODEL_SAVE_PATH}')

    end_time = time.time()
    end_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(end_time))
    print('\nDone with Training at', end_time_str, '!')


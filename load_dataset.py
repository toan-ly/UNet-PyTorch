import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class CarvanaDataset(Dataset):
    def __init__(self, root_path, test=False):
        """
        Args:
            root_path (str): Path to the root directory containing the dataset.
            test (bool): Flag to indicate if the dataset is for testing. Defaults to False.
        """
        self.root_path = root_path
        self.images = sorted([self._load_file_paths('manual_test' if test else 'train')])
        self.masks = sorted([self._load_file_paths('manual_test_masks' if test else 'train_masks')])
            
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        
    def _load_file_paths(self, subdir):
        """
        Load file paths from a specified subdirectory.
        
        Args:
            subdir (str): Subdirectory name.
        
        Returns:
            list: Sorted list of file paths.
        """
        full_dir = os.path.join(self.root_path, subdir)
        return [os.path.join(full_dir, filename) for filename in os.listdir(full_dir)]

    def __getitem__(self, index):
        """
        Get an item from the dataset.
        
        Args:
            index (int): Index of the item.
        
        Returns:
            tuple: Transformed image and mask tensors.
        """
        img_path = self.images[index]
        mask_path = self.masks[index]

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        return self.transform(img), self.transform(mask)

    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: Number of items in the dataset.
        """  
        return len(self.images)
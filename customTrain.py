import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
# import argparse
import matplotlib.pyplot as plt
from colorizers import *
from utils import *
'''TODO: make this file train on the gray images (train_input), and colored images (train_output) '''

class Dataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        # Add any additional initialization steps if needed

    def __len__(self):
        return len(list_of_your_images)

    def __getitem__(self, idx):
        # Load and preprocess the image and label
        img_path = path_to_image_file
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        label = load_corresponding_label(img_path)  # Implement this function

        if self.transform:
            image = self.transform(image)

        return image, label


model = ECCVGenerator()
criterion = nn.MSELoss()

from torch.utils.data import DataLoader

train_dataset = Dataset(train_folder, transform=your_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = Dataset(val_folder, transform=your_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

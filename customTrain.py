import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# import argparse
import matplotlib.pyplot as plt
from colorizers import *
from utils import *
'''TODO: make this file train on the gray images (train_input), and colored images (train_output) '''

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class ColorizationDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.gray_folder = os.path.join(root_folder, 'Gray')
        self.colorful_folder = os.path.join(root_folder, 'ColorfulOriginal')
        self.image_list = os.listdir(self.gray_folder)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Load grayscale image
        gray_image_path = os.path.join(self.gray_folder, self.image_list[idx])
        gray_image = Image.open(gray_image_path).convert("L")

        # Load corresponding colorized image
        colorful_image_path = os.path.join(self.colorful_folder, self.image_list[idx])
        colorful_image = Image.open(colorful_image_path).convert("RGB")

        if self.transform:
            gray_image = self.transform(gray_image)
            colorful_image = self.transform(colorful_image)

        return gray_image, colorful_image
    
    
model = ECCVGenerator()
criterion = nn.MSELoss()

train_dataset = ColorizationDataset(root_folder=train_folder, transform=your_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = ColorizationDataset(root_folder=val_folder, transform=your_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    # Optionally, evaluate the model on the validation set after each epoch

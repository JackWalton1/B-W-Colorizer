from colorizers import *
from utils import *

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
from torchvision.transforms import Grayscale
from PIL import Image
import os
import torch.optim as optim
from torch.nn.functional import interpolate, mse_loss
from skimage import color
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

def custom_collate(batch, target_size=(250, 250)):
    inputs, targets = zip(*batch)

    # Convert inputs and targets to tensors if they are not already
    inputs = [img if isinstance(img, torch.Tensor) else transforms.ToTensor()(img) for img in inputs]
    targets = [img if isinstance(img, torch.Tensor) else transforms.ToTensor()(img) for img in targets]

    # Resize inputs and targets to a common size
    transform = transforms.Compose([
        transforms.Resize(target_size),
    ])

    resized_inputs = [transform(img) for img in inputs]
    resized_targets = [transform(img) for img in targets]

    return torch.stack(resized_inputs), torch.stack(resized_targets)


class ColorizationIterableDataset(IterableDataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.image_list = self.load_image_list()

    def load_image_list(self):
        image_list = []

        # Traverse through each subfolder in the 'Gray' directory
        for folder_name in os.listdir(os.path.join(self.root_folder, 'Gray')):
            gray_folder = os.path.join(self.root_folder, 'Gray', folder_name)
            colorful_folder = os.path.join(self.root_folder, 'ColorfulOriginal', folder_name)

            # Check if the subfolder contains 'Gray' and 'ColorfulOriginal' folders
            if os.path.exists(gray_folder) and os.path.exists(colorful_folder):
                # Get list of images in the 'Gray' folder
                gray_images = os.listdir(gray_folder)

                # Check if the same number of images exist in 'ColorfulOriginal' folder
                if len(gray_images) == len(os.listdir(colorful_folder)):
                    # Add each image pair to the image list
                    for image_name in gray_images:
                        gray_image_path = os.path.join(gray_folder, image_name)
                        colorful_image_path = os.path.join(colorful_folder, image_name)
                        image_list.append((gray_image_path, colorful_image_path))

        print(f"Total images found: {len(image_list)}")
        return image_list

    def __iter__(self):
        for gray_image_path, colorful_image_path in self.image_list:
            # Load grayscale image
            gray_image = Image.open(gray_image_path).convert("L")

            # Load corresponding colorized image
            colorful_image = Image.open(colorful_image_path).convert("RGB")

            # if self.transform:
            gray_image = self.transform(gray_image)

            # Convert colorful image to Lab color space
            colorful_lab = color.rgb2lab(colorful_image)
            # Extract 'a' and 'b' channels
            ab_channels = colorful_lab[:, :, 1:]

            yield gray_image, ab_channels

# Define the transform
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add other transforms if needed
])

model = ECCVGenerator()
criterion = nn.MSELoss()
model.load_state_dict(torch.load('custom_trained_model.pth')) # Can comment this out, its to keep training the saved model weights

# Define the number of worker processes for data loading
# num_workers = 8
# print("\nPerforming training with ", num_workers, " CPUs\n")
train_dataset = ColorizationIterableDataset(root_folder="./train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=30, collate_fn=custom_collate) #, num_workers=num_workers)

num_epochs = 2
log_interval = 10  # Print the training loss every 10 batches
optimizer = optim.Adam(model.parameters()) #, lr=0.001)

target_height, target_width = 250, 250

for epoch in range(num_epochs):
    model.train()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)

        # Resize the output tensor to match the target tensor's size
        outputs_resized = interpolate(outputs, size=(target_height, target_width), mode='bilinear', align_corners=False)

        # Select one channel to simulate a grayscale image
        outputs_resized_grayscale = outputs_resized[:, 0, :, :].unsqueeze(1)

        loss = mse_loss(outputs_resized_grayscale.float(), targets.float())

        loss.backward()
        optimizer.step()

        # if batch_idx % log_interval == 0:
        print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    # Optionally, evaluate the model on the validation set after each epoch
torch.save(model.state_dict(), 'custom_trained_model.pth')

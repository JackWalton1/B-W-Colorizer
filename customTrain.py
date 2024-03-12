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
import torch.optim as optim
from torch.nn.functional import interpolate, mse_loss
from torchvision.transforms import Grayscale

class ColorizationDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.gray_folder = os.path.join(root_folder, 'Gray/Apple')
        self.colorful_folder = os.path.join(root_folder, 'ColorfulOriginal/Apple')
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


# Define the transform and dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add other transforms if needed
])

model = ECCVGenerator()
criterion = nn.MSELoss()

train_dataset = ColorizationDataset(root_folder="./train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True, collate_fn=lambda x: custom_collate(x, target_size=(250, 250)))

# val_dataset = ColorizationDataset(root_folder=val_folder)
# val_loader = DataLoader(val_dataset, batch_size=30, shuffle=False)

import torch.optim as optim
num_epochs = 10
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

        loss = mse_loss(outputs_resized_grayscale, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
            
    # Optionally, evaluate the model on the validation set after each epoch
torch.save(model.state_dict(), 'custom_trained_model.pth')

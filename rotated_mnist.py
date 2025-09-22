import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random
import math

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5,), (0.5,))
])

full_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
full_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def sample_subset(dataset, percent=0.1):
   class_indices = {i: [] for i in range(10)}
   for idx, (_, label) in enumerate(dataset):
       class_indices[label].append(idx)

   selected_indices = []
   for class_idx in range(10):
       n_samples = int(len(class_indices[class_idx]) * percent)
       selected_indices.extend(random.sample(class_indices[class_idx], n_samples))

   return torch.utils.data.Subset(dataset, selected_indices)

train_subset = sample_subset(full_train, 1.0)
test_subset = full_test


train_size = int(0.8 * len(train_subset))
val_size = len(train_subset) - train_size
train_data, val_data = torch.utils.data.random_split(train_subset, [train_size, val_size])

print(f"Train samples: {len(train_data)}")
print(f"Val samples: {len(val_data)}")
print(f"Test samples: {len(test_subset)}")


def rotate_image(image, angle):
    discrete_angle = round(angle / 90) * 90

    if discrete_angle % 360 == 0:
        return image  , 0
    elif discrete_angle % 360 == 90:
        return torch.rot90(image, k=1, dims=[-2, -1]) , 90
    elif discrete_angle % 360 == 180:
        return torch.rot90(image, k=2, dims=[-2, -1]) ,180
    elif discrete_angle % 360 == 270:
        return torch.rot90(image, k=3, dims=[-2, -1]) , 270
    else:
        return image , 0


class PairedRotationDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        angle = random.uniform(0, 360)
        rotated_image ,angle_actual = rotate_image(image, angle)

        return image, rotated_image, angle_actual, label


train_paired = PairedRotationDataset(train_data)
val_paired = PairedRotationDataset(val_data)
test_paired = PairedRotationDataset(test_subset)

train_loader = DataLoader(train_paired, batch_size=64, shuffle=True)
val_loader = DataLoader(val_paired, batch_size=64, shuffle=False)
test_loader = DataLoader(test_paired, batch_size=64, shuffle=False)

def rotate_latent(z_v, angles):
    z_v_rot = []
    for i, angle in enumerate(angles):
        discrete_angle = round(angle.item() / 90) * 90 % 360
        if discrete_angle == 0:
            z_v_rot.append(z_v[i])
        elif discrete_angle == 90:
            z_v_rot.append(torch.rot90(z_v[i], k=1, dims=[-2,-1]))
        elif discrete_angle == 180:
            z_v_rot.append(torch.rot90(z_v[i], k=2, dims=[-2,-1]))
        elif discrete_angle == 270:
            z_v_rot.append(torch.rot90(z_v[i], k=3, dims=[-2,-1]))
    return torch.stack(z_v_rot, dim=0)

class ConvEncoder(nn.Module):
    def __init__(self, latent_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, latent_channels, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.encoder(x)


class ConvDecoder(nn.Module):
    def __init__(self, latent_channels=64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.decoder(z)

class ChannelMaskSTE(nn.Module):
    def __init__(self, latent_channels, threshold=0.5):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(latent_channels) * 0.1)
        self.threshold = threshold

    def forward(self, z):
        probs = torch.sigmoid(self.logits)
        hard_mask = (probs > self.threshold).float()
        mask = hard_mask + (probs - probs.detach())
        mask = mask.view(1, -1, 1, 1)
        return mask



class GroupLatentAutoencoder(nn.Module):
    def __init__(self, latent_channels=64, threshold=0.5):
        super().__init__()
        self.encoder = ConvEncoder(latent_channels)
        self.decoder = ConvDecoder(latent_channels)
        self.mask_module = ChannelMaskSTE(latent_channels, threshold)

    def forward_with_latent_rotation(self, x, angles):
        z = self.encoder(x)
        mask = self.mask_module(z)
        z_v = mask * z
        z_i = (1 - mask) * z
        z_v_rot = rotate_latent(z_v, angles)
        z_comb_rot = z_i + z_v_rot
        recon_rot = self.decoder(z_comb_rot)

        return {
            'z': z,
            'mask': mask,
            'z_v': z_v,
            'z_i': z_i,
            'z_v_rot': z_v_rot,
            'recon_rot': recon_rot,
        }

latent_channels = 64
lr = 1e-3
epochs = 50
threshold_mask = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lm_inv =1.0
lm_const=1.0


model = GroupLatentAutoencoder(latent_channels=latent_channels, threshold=threshold_mask).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_losses, val_losses = [], []
best_val = float('inf')


def train_epoch(model, dataloader, optimizer, epoch):
    model.train()
    total_loss = 0

    for batch_idx, (x1, x2, angles, _) in enumerate(dataloader):
        x1, x2 = x1.to(device), x2.to(device)
        angles = angles.to(device)

        optimizer.zero_grad()
        outputs = model.forward_with_latent_rotation(x1, angles)
        recon_loss_rot   = F.mse_loss(outputs['recon_rot'], x2)
        with torch.no_grad():
            outputs_x2 = model.forward_with_latent_rotation(x2, angles)
            z_i2 = outputs_x2['z_i']
            z_v2 = outputs_x2['z_v']
        loss_invariant = F.mse_loss(outputs['z_i'], z_i2)
        latent_v_loss = F.mse_loss(outputs['z_v_rot'], z_v2)

        loss = (  recon_loss_rot
                + lm_inv * loss_invariant
                + lm_const * latent_v_loss
               )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    train_losses.append(avg_loss)
    print(f"Train Loss: {avg_loss:.4f}")


def validate_epoch(model, dataloader, epoch):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (x1, x2, angles, _) in enumerate(dataloader):
            x1, x2 = x1.to(device), x2.to(device)
            angles = angles.to(device)
            outputs = model.forward_with_latent_rotation(x1, angles)
            recon_loss_rot   = F.mse_loss(outputs['recon_rot'], x2)
            outputs_x2 = model.forward_with_latent_rotation(x2, angles)
            loss_invariant = F.mse_loss(outputs['z_i'], outputs_x2['z_i'])
            latent_v_loss  = F.mse_loss(outputs['z_v_rot'], outputs_x2['z_v'])

            loss = (  recon_loss_rot
                    + lm_inv * loss_invariant
                    + lm_const * latent_v_loss
                  )

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    val_losses.append(avg_loss)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

best_val = float('inf')

for epoch in range(epochs):
    print(f"\n=== Epoch {epoch+1}/{epochs} ===")
    train_epoch(model, train_loader, optimizer, epoch)
    val_loss = validate_epoch(model, val_loader, epoch)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), 'rotated_model.pth')
        print(f"Saved best model (val_loss: {best_val:.4f})")

print("Training completed.")

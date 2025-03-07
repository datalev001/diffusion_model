########traditional diffussion without AR#########
import gzip
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F

###########################################
# 1. Load MNIST Images from Gzipped File
###########################################
def load_mnist_images(path):
    """Load MNIST images from a gzipped file."""
    with gzip.open(path, 'rb') as f:
        f.read(16)  # Skip header
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(-1, 28, 28)
    return data

# Update to your local MNIST gz file path
mnist_path = "train-images-idx3-ubyte.gz"
mnist_images = load_mnist_images(mnist_path)
print(f"Loaded MNIST images: {mnist_images.shape}")

###########################################
# 2. Create MNIST Dataset Class
###########################################
class MNISTDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx], mode='L')
        if self.transform:
            img = self.transform(img)
        return img

# Define dataset transforms (normalize to [-1,1] for diffusion)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
mnist_dataset = MNISTDataset(mnist_images, transform=transform)
mnist_loader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)

###########################################
# 3. Diffusion Model (Fixed U-Net)
###########################################
class SimpleUNetMNIST(nn.Module):
    """U-Net model for MNIST diffusion."""
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)  # (base, 14,14)
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)  # (2*base, 7,7)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(base_channels * 2 + base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(base_channels, in_channels, kernel_size=2, stride=2)
        self.conv_up2 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        bn = self.bottleneck(p2)
        up1 = self.up1(bn)
        up1 = self.conv_up1(torch.cat([up1, d2], dim=1))
        up2 = self.up2(up1)
        out = self.conv_up2(torch.cat([up2, x], dim=1))
        return out

###########################################
# 4. Traditional Diffusion Framework
###########################################
def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)

class DiffusionFramework:
    """
    Standard diffusion training:
      - Sample a random timestep t,
      - Add pure Gaussian noise to the image,
      - Train the model to predict this noise.
    """
    def __init__(self, model, timesteps=200, device="cpu"):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device
        self.betas = linear_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0).to(device)
    
    def train_step(self, x0):
        b = x0.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=x0.device).long()
        noise = torch.randn_like(x0)
        alpha_hat_t = self.alpha_hats[t].reshape(b, 1, 1, 1)
        xt = torch.sqrt(alpha_hat_t) * x0 + torch.sqrt(1.0 - alpha_hat_t) * noise
        predicted_noise = self.model(xt)
        loss = nn.functional.mse_loss(predicted_noise, noise)
        return loss
    
    @torch.no_grad()
    def p_sample(self, xt, t):
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_hat_t = self.alpha_hats[t]
        eps = self.model(xt)
        coef = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_hat_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (xt - coef * eps)
        if t > 0:
            z = torch.randn_like(xt)
            sigma_t = torch.sqrt(beta_t)
            sample = mean + sigma_t * z
        else:
            sample = mean
        return sample
    
    @torch.no_grad()
    def p_sample_loop(self, x_T):
        xt = x_T
        for i in reversed(range(self.timesteps)):
            xt = self.p_sample(xt, i)
        return xt.clamp(-1, 1)

###########################################
# 5. Training and Sampling with Multiple Example Outputs
###########################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the diffusion model (traditional diffusion framework)
    model_diffusion = SimpleUNetMNIST().to(device)
    diffusion = DiffusionFramework(model_diffusion, timesteps=200, device=device)
    optimizer_diff = optim.Adam(diffusion.model.parameters(), lr=1e-4)

    print("Training Diffusion Model...")
    for epoch in range(60):
        total_loss = 0
        for real_imgs in mnist_loader:
            real_imgs = real_imgs.to(device)
            loss = diffusion.train_step(real_imgs)
            optimizer_diff.zero_grad()
            loss.backward()
            optimizer_diff.step()
            total_loss += loss.item()
        print(f"Diffusion Epoch {epoch+1}/60, Loss: {total_loss/len(mnist_loader):.4f}")

    # Generate multiple examples for comparison.
    diffusion.model.eval()
    num_examples = 5  # Number of examples to display
    originals = []
    diffusion_outputs = []

    # Get one batch of original samples.
    batch = next(iter(mnist_loader))
    for i in range(num_examples):
        # Original image.
        originals.append(batch[i].cpu())
        # Generate diffusion output using standard diffusion (starting from pure Gaussian noise).
        x_T = torch.randn(1, 1, 28, 28, device=device)
        diff_out = diffusion.p_sample_loop(x_T)
        diffusion_outputs.append(diff_out.squeeze(0).cpu())

    # Plot the results in a grid: each row shows Original and Diffusion output.
    fig, axes = plt.subplots(num_examples, 2, figsize=(8, 3*num_examples))
    for i in range(num_examples):
        axes[i, 0].imshow(originals[i].squeeze().numpy(), cmap="gray")
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(diffusion_outputs[i].squeeze().numpy(), cmap="gray")
        axes[i, 1].set_title("Diffusion Output")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

Loaded MNIST images: (60000, 28, 28)
Training Diffusion Model...
Diffusion Epoch 1/60, Loss: 0.5296
Diffusion Epoch 2/60, Loss: 0.3034
Diffusion Epoch 3/60, Loss: 0.2428
Diffusion Epoch 4/60, Loss: 0.2070
Diffusion Epoch 5/60, Loss: 0.1834
Diffusion Epoch 6/60, Loss: 0.1687
Diffusion Epoch 7/60, Loss: 0.1568
Diffusion Epoch 8/60, Loss: 0.1498
Diffusion Epoch 9/60, Loss: 0.1440
Diffusion Epoch 10/60, Loss: 0.1379
Diffusion Epoch 11/60, Loss: 0.1354
Diffusion Epoch 12/60, Loss: 0.1322
Diffusion Epoch 13/60, Loss: 0.1296
Diffusion Epoch 14/60, Loss: 0.1263
Diffusion Epoch 15/60, Loss: 0.1234
Diffusion Epoch 16/60, Loss: 0.1214
Diffusion Epoch 17/60, Loss: 0.1201
Diffusion Epoch 18/60, Loss: 0.1191
Diffusion Epoch 19/60, Loss: 0.1175
Diffusion Epoch 20/60, Loss: 0.1153
Diffusion Epoch 21/60, Loss: 0.1138
Diffusion Epoch 22/60, Loss: 0.1125
Diffusion Epoch 23/60, Loss: 0.1119
Diffusion Epoch 24/60, Loss: 0.1111
Diffusion Epoch 25/60, Loss: 0.1091
Diffusion Epoch 26/60, Loss: 0.1089
Diffusion Epoch 27/60, Loss: 0.1083
Diffusion Epoch 28/60, Loss: 0.1067
Diffusion Epoch 29/60, Loss: 0.1059
Diffusion Epoch 30/60, Loss: 0.1055
Diffusion Epoch 31/60, Loss: 0.1048
Diffusion Epoch 32/60, Loss: 0.1035
Diffusion Epoch 33/60, Loss: 0.1037
Diffusion Epoch 34/60, Loss: 0.1020
Diffusion Epoch 35/60, Loss: 0.1015
Diffusion Epoch 36/60, Loss: 0.1017
Diffusion Epoch 37/60, Loss: 0.1003
Diffusion Epoch 38/60, Loss: 0.0999
Diffusion Epoch 39/60, Loss: 0.0996
Diffusion Epoch 40/60, Loss: 0.0986
Diffusion Epoch 41/60, Loss: 0.0981
Diffusion Epoch 42/60, Loss: 0.0978
Diffusion Epoch 43/60, Loss: 0.0972
Diffusion Epoch 44/60, Loss: 0.0968
Diffusion Epoch 45/60, Loss: 0.0964
Diffusion Epoch 46/60, Loss: 0.0956
Diffusion Epoch 47/60, Loss: 0.0953
Diffusion Epoch 48/60, Loss: 0.0952
Diffusion Epoch 49/60, Loss: 0.0950
Diffusion Epoch 50/60, Loss: 0.0943
Diffusion Epoch 51/60, Loss: 0.0943
Diffusion Epoch 52/60, Loss: 0.0932
Diffusion Epoch 53/60, Loss: 0.0935
Diffusion Epoch 54/60, Loss: 0.0929
Diffusion Epoch 55/60, Loss: 0.0927
Diffusion Epoch 56/60, Loss: 0.0920
Diffusion Epoch 57/60, Loss: 0.0919
Diffusion Epoch 58/60, Loss: 0.0913
Diffusion Epoch 59/60, Loss: 0.0910
Diffusion Epoch 60/60, Loss: 0.0906

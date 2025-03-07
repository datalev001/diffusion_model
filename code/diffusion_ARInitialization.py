#########method 2: AR Prior Initialization Diffusion (Method 2)################
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

# Define dataset transforms (normalizing to [-1,1] for diffusion)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
mnist_dataset = MNISTDataset(mnist_images, transform=transform)
mnist_loader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)

###########################################
# 3. Autoregressive Prior (LSTM-based)
###########################################
class ARPrior(nn.Module):
    """Autoregressive Prior using LSTM to learn structured noise."""
    def __init__(self, input_dim=100, img_dim=28*28, hidden_dim=512):
        super().__init__()
        self.img_dim = img_dim  # 28x28 = 784
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, img_dim)

    def forward(self, z):
        # z shape: (batch, 100)
        z = z.unsqueeze(1)  # -> (batch, 1, 100)
        lstm_out, _ = self.lstm(z)
        out = self.fc(lstm_out[:, -1, :])  # last time-step
        batch_size = out.shape[0]
        return out.view(batch_size, 1, 28, 28)

###########################################
# 4. Diffusion Model (U-Net)
###########################################
class SimpleUNetMNIST(nn.Module):
    """U-Net model for MNIST diffusion."""
    def __init__(self, in_channels=1, base_channels=64):
        """
        Using base_channels=64 for higher capacity.
        """
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
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)  # (2*base, 7,7)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        # Concatenation: channels become (base_channels*2 + base_channels)
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(base_channels*2 + base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(base_channels, in_channels, kernel_size=2, stride=2)
        self.conv_up2 = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)

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
# 5. Standard Diffusion Framework (no AR consistency loss)
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
    def p_sample_loop_from(self, x_init):
        xt = x_init.clone()
        for i in reversed(range(self.timesteps)):
            xt = self.p_sample(xt, i)
        return xt.clamp(-1, 1)
    
    @torch.no_grad()
    def p_sample_loop(self, shape):
        xt = torch.randn(shape, device=self.device)
        for i in reversed(range(self.timesteps)):
            xt = self.p_sample(xt, i)
        return xt.clamp(-1, 1)

###########################################
# 6. Joint Training and Sampling (Standard Diffusion with AR Prior Blending at Inference)
###########################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 6.1 Train AR Prior separately
    ar_prior = ARPrior().to(device)
    optimizer_ar = optim.Adam(ar_prior.parameters(), lr=1e-4)
    num_ar_epochs = 10
    print("Training AR Prior...")
    for epoch in range(num_ar_epochs):
        total_loss_ar = 0.0
        for real_imgs in mnist_loader:
            real_imgs = real_imgs.to(device)
            real_flat = real_imgs.view(real_imgs.size(0), -1)
            z = torch.randn(real_imgs.size(0), 100, device=device)
            out = ar_prior(z).view(real_imgs.size(0), -1)
            loss = nn.functional.mse_loss(out, real_flat)
            optimizer_ar.zero_grad()
            loss.backward()
            optimizer_ar.step()
            total_loss_ar += loss.item()
        avg_loss_ar = total_loss_ar / len(mnist_loader)
        print(f"AR Prior Epoch {epoch+1}/{num_ar_epochs}, Loss: {avg_loss_ar:.4f}")
    # Freeze AR Prior for inference.
    for param in ar_prior.parameters():
        param.requires_grad = False
    
    # 6.2 Train Diffusion Model (Standard Diffusion)
    model_diffusion = SimpleUNetMNIST().to(device)
    diffusion = DiffusionFramework(model_diffusion, timesteps=200, device=device)
    optimizer_diff = optim.Adam(diffusion.model.parameters(), lr=1e-4)
    num_diff_epochs = 60
    print("\nTraining Diffusion Model...")
    for epoch in range(num_diff_epochs):
        total_loss_diff = 0.0
        for real_imgs in mnist_loader:
            real_imgs = real_imgs.to(device)
            loss = diffusion.train_step(real_imgs)
            optimizer_diff.zero_grad()
            loss.backward()
            optimizer_diff.step()
            total_loss_diff += loss.item()
        avg_loss_diff = total_loss_diff / len(mnist_loader)
        print(f"Diffusion Epoch {epoch+1}/{num_diff_epochs}, Loss: {avg_loss_diff:.4f}")
    
    # 6.3 Generate multiple examples for comparison.
    diffusion.model.eval()
    ar_prior.eval()
    num_examples = 5  # Number of examples to display
    originals = []
    ar_outputs = []
    diffusion_outputs = []
    
    # Get a batch of original samples.
    batch = next(iter(mnist_loader))
    for i in range(num_examples):
        # Original image
        originals.append(batch[i].cpu())
        # Generate AR prior output for one example:
        z = torch.randn(1, 100, device=device)
        x0_ar = ar_prior(z)
        ar_outputs.append(x0_ar.squeeze(0).cpu())
        # Generate final diffusion output using AR prior blending at inference:
        # Here we blend AR output with noise:
        alpha_blend = 0.1  # Adjust blending factor as needed.
        xT_gauss = torch.randn_like(x0_ar)
        xT_init = alpha_blend * x0_ar + (1 - alpha_blend) * xT_gauss
        gen_sample = diffusion.p_sample_loop_from(xT_init)
        diffusion_outputs.append(gen_sample.squeeze(0).cpu())
    
    # 6.4 Plot a grid: Each row shows Original, AR Prior, and Diffusion+AR output.
    fig, axes = plt.subplots(num_examples, 3, figsize=(9, 3*num_examples))
    for i in range(num_examples):
        axes[i, 0].imshow(originals[i].squeeze().numpy(), cmap="gray")
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(ar_outputs[i].squeeze().numpy(), cmap="gray")
        axes[i, 1].set_title("AR Prior")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(diffusion_outputs[i].squeeze().numpy(), cmap="gray")
        axes[i, 2].set_title("Diffusion + AR")
        axes[i, 2].axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

##############################################
Loaded MNIST images: (60000, 28, 28)
Training AR Prior...
AR Prior Epoch 1/10, Loss: 0.3384
AR Prior Epoch 2/10, Loss: 0.2698
AR Prior Epoch 3/10, Loss: 0.2696
AR Prior Epoch 4/10, Loss: 0.2695
AR Prior Epoch 5/10, Loss: 0.2694
AR Prior Epoch 6/10, Loss: 0.2694
AR Prior Epoch 7/10, Loss: 0.2694
AR Prior Epoch 8/10, Loss: 0.2693
AR Prior Epoch 9/10, Loss: 0.2693
AR Prior Epoch 10/10, Loss: 0.2693

Training Diffusion Model...
Diffusion Epoch 1/60, Loss: 0.4971
Diffusion Epoch 2/60, Loss: 0.2690
Diffusion Epoch 3/60, Loss: 0.2173
Diffusion Epoch 4/60, Loss: 0.1880
Diffusion Epoch 5/60, Loss: 0.1682
Diffusion Epoch 6/60, Loss: 0.1520
Diffusion Epoch 7/60, Loss: 0.1392
Diffusion Epoch 8/60, Loss: 0.1293
Diffusion Epoch 9/60, Loss: 0.1200
Diffusion Epoch 10/60, Loss: 0.1155
Diffusion Epoch 11/60, Loss: 0.1107
Diffusion Epoch 12/60, Loss: 0.1068
Diffusion Epoch 13/60, Loss: 0.1040
Diffusion Epoch 14/60, Loss: 0.1016
Diffusion Epoch 15/60, Loss: 0.0988
Diffusion Epoch 16/60, Loss: 0.0965
Diffusion Epoch 17/60, Loss: 0.0937
Diffusion Epoch 18/60, Loss: 0.0931
Diffusion Epoch 19/60, Loss: 0.0908
Diffusion Epoch 20/60, Loss: 0.0896
Diffusion Epoch 21/60, Loss: 0.0886
Diffusion Epoch 22/60, Loss: 0.0873
Diffusion Epoch 23/60, Loss: 0.0864
Diffusion Epoch 24/60, Loss: 0.0856
Diffusion Epoch 25/60, Loss: 0.0844
Diffusion Epoch 26/60, Loss: 0.0838
Diffusion Epoch 27/60, Loss: 0.0828
Diffusion Epoch 28/60, Loss: 0.0816
Diffusion Epoch 29/60, Loss: 0.0817
Diffusion Epoch 30/60, Loss: 0.0801
Diffusion Epoch 31/60, Loss: 0.0803
Diffusion Epoch 32/60, Loss: 0.0788
Diffusion Epoch 33/60, Loss: 0.0783
Diffusion Epoch 34/60, Loss: 0.0786
Diffusion Epoch 35/60, Loss: 0.0774
Diffusion Epoch 36/60, Loss: 0.0768
Diffusion Epoch 37/60, Loss: 0.0770
Diffusion Epoch 38/60, Loss: 0.0755
Diffusion Epoch 39/60, Loss: 0.0756
Diffusion Epoch 40/60, Loss: 0.0750
Diffusion Epoch 41/60, Loss: 0.0748
Diffusion Epoch 42/60, Loss: 0.0742
Diffusion Epoch 43/60, Loss: 0.0732
Diffusion Epoch 44/60, Loss: 0.0736
Diffusion Epoch 45/60, Loss: 0.0731
Diffusion Epoch 46/60, Loss: 0.0724
Diffusion Epoch 47/60, Loss: 0.0726
Diffusion Epoch 48/60, Loss: 0.0721
Diffusion Epoch 49/60, Loss: 0.0718
Diffusion Epoch 50/60, Loss: 0.0711
Diffusion Epoch 51/60, Loss: 0.0709
Diffusion Epoch 52/60, Loss: 0.0708
Diffusion Epoch 53/60, Loss: 0.0704
Diffusion Epoch 54/60, Loss: 0.0698
Diffusion Epoch 55/60, Loss: 0.0694
Diffusion Epoch 56/60, Loss: 0.0698
Diffusion Epoch 57/60, Loss: 0.0690
Diffusion Epoch 58/60, Loss: 0.0689
Diffusion Epoch 59/60, Loss: 0.0682
Diffusion Epoch 60/60, Loss: 0.0682

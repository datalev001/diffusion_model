###############AR Consistency Loss Diffusion (Method 1):λ AR ################################
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
        img = Image.fromarray(self.images[idx], mode='L')  # Grayscale PIL image
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
        out = self.fc(lstm_out[:, -1, :])  # use last timestep
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
# 5. Diffusion Framework with AR Consistency Loss
###########################################
def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)

class DiffusionFramework:
    """
    Standard diffusion training combined with an AR consistency loss.
    Diffusion Loss (L_diff): Predict noise from a noisy image.
    AR Consistency Loss (L_AR): Encourage the denoised output x0_pred
      to be close to an AR prior sample.
    
    Total Loss: L = L_diff + lambda_ar * L_AR
    """
    def __init__(self, model, timesteps=200, device="cpu"):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device
        self.betas = linear_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0).to(device)
    
    def train_step(self, x0, ar_sample, lambda_ar=0.2):
        """
        x0: real image (batch, 1, 28, 28)
        ar_sample: AR prior sample (batch, 1, 28, 28) from a pre-trained/frozen AR model
        lambda_ar: weight for the AR consistency loss.
        """
        b = x0.size(0)
        # Sample random timesteps for each image.
        t = torch.randint(0, self.timesteps, (b,), device=x0.device).long()
        noise = torch.randn_like(x0)
        alpha_hat_t = self.alpha_hats[t].reshape(b, 1, 1, 1)

        # Forward diffusion: add noise
        xt = torch.sqrt(alpha_hat_t) * x0 + torch.sqrt(1.0 - alpha_hat_t) * noise
        
        # Diffusion model predicts the noise:
        predicted_noise = self.model(xt)
        loss_diff = nn.functional.mse_loss(predicted_noise, noise)
        
        # Compute predicted x0 from noise prediction:
        x0_pred = (xt - torch.sqrt(1.0 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_hat_t)
        
        # AR consistency loss:
        loss_ar = nn.functional.mse_loss(x0_pred, ar_sample)
        
        total_loss = loss_diff + lambda_ar * loss_ar
        return total_loss, loss_diff, loss_ar
    
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
        """Reverse diffusion starting from an initial state (x_init)."""
        xt = x_init.clone()
        for i in reversed(range(self.timesteps)):
            xt = self.p_sample(xt, i)
        return xt.clamp(-1, 1)
    
    @torch.no_grad()
    def p_sample_loop(self, shape):
        """Standard reverse diffusion starting from pure noise."""
        xt = torch.randn(shape, device=self.device)
        for i in reversed(range(self.timesteps)):
            xt = self.p_sample(xt, i)
        return xt.clamp(-1, 1)

###########################################
# 6. Joint Training and Sampling with AR Consistency Loss
###########################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 6.1 Train the AR Prior separately (or assume pre-trained)
    ar_prior = ARPrior().to(device)
    optimizer_ar = optim.Adam(ar_prior.parameters(), lr=1e-4)
    num_ar_epochs = 10
    print("Training AR Prior...")
    for epoch in range(num_ar_epochs):
        total_loss_ar = 0.0
        for real_imgs in mnist_loader:
            real_imgs = real_imgs.to(device)
            z = torch.randn(real_imgs.size(0), 100, device=device)
            out = ar_prior(z)
            loss = nn.functional.mse_loss(out, real_imgs)
            optimizer_ar.zero_grad()
            loss.backward()
            optimizer_ar.step()
            total_loss_ar += loss.item()
        avg_loss = total_loss_ar / len(mnist_loader)
        print(f"AR Prior Epoch {epoch+1}/{num_ar_epochs}, Loss: {avg_loss:.4f}")
    # Freeze AR prior parameters for diffusion training.
    for param in ar_prior.parameters():
        param.requires_grad = False
    
    # 6.2 Train the Diffusion Model with AR consistency loss.
    model_diffusion = SimpleUNetMNIST().to(device)
    diffusion = DiffusionFramework(model_diffusion, timesteps=200, device=device)
    optimizer_diff = optim.Adam(diffusion.model.parameters(), lr=1e-4)
    num_diff_epochs = 60
    lambda_ar = 0.2
    print("\nTraining Diffusion Model with AR consistency loss...")
    for epoch in range(num_diff_epochs):
        total_loss = 0.0
        total_loss_diff = 0.0
        total_loss_ar = 0.0
        for real_imgs in mnist_loader:
            real_imgs = real_imgs.to(device)
            b = real_imgs.size(0)
            z = torch.randn(b, 100, device=device)
            x0_ar = ar_prior(z)  # AR prior sample
            loss, loss_diff, loss_ar = diffusion.train_step(real_imgs, x0_ar, lambda_ar=lambda_ar)
            optimizer_diff.zero_grad()
            loss.backward()
            optimizer_diff.step()
            total_loss += loss.item()
            total_loss_diff += loss_diff.item()
            total_loss_ar += loss_ar.item()
        avg_loss = total_loss / len(mnist_loader)
        avg_loss_diff = total_loss_diff / len(mnist_loader)
        avg_loss_ar = total_loss_ar / len(mnist_loader)
        print(f"Diffusion Epoch {epoch+1}/{num_diff_epochs}, Total Loss: {avg_loss:.4f}, Diff Loss: {avg_loss_diff:.4f}, AR Loss: {avg_loss_ar:.4f}")
    
    # 6.3 Generate Multiple Samples for Comparison
    diffusion.model.eval()
    ar_prior.eval()
    num_examples = 5
    originals = []
    ar_outputs = []
    generated = []
    
    # Get a batch of original samples:
    batch = next(iter(mnist_loader))
    for i in range(num_examples):
        originals.append(batch[i].cpu())
        
        # Generate AR prior output:
        z = torch.randn(1, 100, device=device)
        x0_ar = ar_prior(z)
        ar_outputs.append(x0_ar.squeeze(0).cpu())
        
        # Generate final sample using diffusion (using pure AR output as initialization)
        xT_init = x0_ar.clone()
        gen_sample = diffusion.p_sample_loop_from(xT_init)
        generated.append(gen_sample.squeeze(0).cpu())
    
    # 6.4 Plot a grid of examples: Each row shows Original, AR prior, and Diffusion output.
    fig, axes = plt.subplots(num_examples, 3, figsize=(9, 3*num_examples))
    for i in range(num_examples):
        axes[i, 0].imshow(originals[i].squeeze().numpy(), cmap="gray")
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(ar_outputs[i].squeeze().numpy(), cmap="gray")
        axes[i, 1].set_title("AR Prior")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(generated[i].squeeze().numpy(), cmap="gray")
        axes[i, 2].set_title("Diffusion + AR")
        axes[i, 2].axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

####################################################
Loaded MNIST images: (60000, 28, 28)
Training AR Prior...
AR Prior Epoch 1/10, Loss: 0.3390
AR Prior Epoch 2/10, Loss: 0.2698
AR Prior Epoch 3/10, Loss: 0.2696
AR Prior Epoch 4/10, Loss: 0.2695
AR Prior Epoch 5/10, Loss: 0.2694
AR Prior Epoch 6/10, Loss: 0.2694
AR Prior Epoch 7/10, Loss: 0.2694
AR Prior Epoch 8/10, Loss: 0.2693
AR Prior Epoch 9/10, Loss: 0.2693
AR Prior Epoch 10/10, Loss: 0.2693

Training Diffusion Model with AR consistency loss...
Diffusion Epoch 1/60, Total Loss: 0.6107, Diff Loss: 0.4860, AR Loss: 0.6236
Diffusion Epoch 2/60, Total Loss: 0.3097, Diff Loss: 0.2520, AR Loss: 0.2883
Diffusion Epoch 3/60, Total Loss: 0.2499, Diff Loss: 0.2016, AR Loss: 0.2417
Diffusion Epoch 4/60, Total Loss: 0.2206, Diff Loss: 0.1762, AR Loss: 0.2219
Diffusion Epoch 5/60, Total Loss: 0.2002, Diff Loss: 0.1584, AR Loss: 0.2092
Diffusion Epoch 6/60, Total Loss: 0.1874, Diff Loss: 0.1471, AR Loss: 0.2012
Diffusion Epoch 7/60, Total Loss: 0.1777, Diff Loss: 0.1387, AR Loss: 0.1952
Diffusion Epoch 8/60, Total Loss: 0.1702, Diff Loss: 0.1319, AR Loss: 0.1913
Diffusion Epoch 9/60, Total Loss: 0.1644, Diff Loss: 0.1267, AR Loss: 0.1884
Diffusion Epoch 10/60, Total Loss: 0.1578, Diff Loss: 0.1207, AR Loss: 0.1856
Diffusion Epoch 11/60, Total Loss: 0.1533, Diff Loss: 0.1166, AR Loss: 0.1838
Diffusion Epoch 12/60, Total Loss: 0.1492, Diff Loss: 0.1127, AR Loss: 0.1822
Diffusion Epoch 13/60, Total Loss: 0.1459, Diff Loss: 0.1096, AR Loss: 0.1813
Diffusion Epoch 14/60, Total Loss: 0.1432, Diff Loss: 0.1071, AR Loss: 0.1802
Diffusion Epoch 15/60, Total Loss: 0.1405, Diff Loss: 0.1047, AR Loss: 0.1791
Diffusion Epoch 16/60, Total Loss: 0.1389, Diff Loss: 0.1031, AR Loss: 0.1790
Diffusion Epoch 17/60, Total Loss: 0.1367, Diff Loss: 0.1010, AR Loss: 0.1784
Diffusion Epoch 18/60, Total Loss: 0.1351, Diff Loss: 0.0995, AR Loss: 0.1780
Diffusion Epoch 19/60, Total Loss: 0.1340, Diff Loss: 0.0985, AR Loss: 0.1776
Diffusion Epoch 20/60, Total Loss: 0.1329, Diff Loss: 0.0975, AR Loss: 0.1774
Diffusion Epoch 21/60, Total Loss: 0.1303, Diff Loss: 0.0951, AR Loss: 0.1765
Diffusion Epoch 22/60, Total Loss: 0.1298, Diff Loss: 0.0945, AR Loss: 0.1765
Diffusion Epoch 23/60, Total Loss: 0.1292, Diff Loss: 0.0938, AR Loss: 0.1772
Diffusion Epoch 24/60, Total Loss: 0.1278, Diff Loss: 0.0925, AR Loss: 0.1763
Diffusion Epoch 25/60, Total Loss: 0.1264, Diff Loss: 0.0911, AR Loss: 0.1764
Diffusion Epoch 26/60, Total Loss: 0.1259, Diff Loss: 0.0906, AR Loss: 0.1765
Diffusion Epoch 27/60, Total Loss: 0.1248, Diff Loss: 0.0896, AR Loss: 0.1764
Diffusion Epoch 28/60, Total Loss: 0.1241, Diff Loss: 0.0889, AR Loss: 0.1760
Diffusion Epoch 29/60, Total Loss: 0.1226, Diff Loss: 0.0874, AR Loss: 0.1763
Diffusion Epoch 30/60, Total Loss: 0.1213, Diff Loss: 0.0862, AR Loss: 0.1757
Diffusion Epoch 31/60, Total Loss: 0.1203, Diff Loss: 0.0852, AR Loss: 0.1752
Diffusion Epoch 32/60, Total Loss: 0.1198, Diff Loss: 0.0848, AR Loss: 0.1754
Diffusion Epoch 33/60, Total Loss: 0.1190, Diff Loss: 0.0839, AR Loss: 0.1756
Diffusion Epoch 34/60, Total Loss: 0.1183, Diff Loss: 0.0833, AR Loss: 0.1752
Diffusion Epoch 35/60, Total Loss: 0.1180, Diff Loss: 0.0828, AR Loss: 0.1760
Diffusion Epoch 36/60, Total Loss: 0.1170, Diff Loss: 0.0819, AR Loss: 0.1753
Diffusion Epoch 37/60, Total Loss: 0.1164, Diff Loss: 0.0813, AR Loss: 0.1752
Diffusion Epoch 38/60, Total Loss: 0.1169, Diff Loss: 0.0816, AR Loss: 0.1763
Diffusion Epoch 39/60, Total Loss: 0.1158, Diff Loss: 0.0807, AR Loss: 0.1759
Diffusion Epoch 40/60, Total Loss: 0.1150, Diff Loss: 0.0799, AR Loss: 0.1754
Diffusion Epoch 41/60, Total Loss: 0.1139, Diff Loss: 0.0791, AR Loss: 0.1741
Diffusion Epoch 42/60, Total Loss: 0.1143, Diff Loss: 0.0793, AR Loss: 0.1751
Diffusion Epoch 43/60, Total Loss: 0.1139, Diff Loss: 0.0789, AR Loss: 0.1752
Diffusion Epoch 44/60, Total Loss: 0.1132, Diff Loss: 0.0781, AR Loss: 0.1755
Diffusion Epoch 45/60, Total Loss: 0.1134, Diff Loss: 0.0783, AR Loss: 0.1755
Diffusion Epoch 46/60, Total Loss: 0.1130, Diff Loss: 0.0779, AR Loss: 0.1757
Diffusion Epoch 47/60, Total Loss: 0.1124, Diff Loss: 0.0773, AR Loss: 0.1754
Diffusion Epoch 48/60, Total Loss: 0.1120, Diff Loss: 0.0771, AR Loss: 0.1748
Diffusion Epoch 49/60, Total Loss: 0.1121, Diff Loss: 0.0770, AR Loss: 0.1754
Diffusion Epoch 50/60, Total Loss: 0.1111, Diff Loss: 0.0762, AR Loss: 0.1749
Diffusion Epoch 51/60, Total Loss: 0.1106, Diff Loss: 0.0756, AR Loss: 0.1754
Diffusion Epoch 52/60, Total Loss: 0.1108, Diff Loss: 0.0757, AR Loss: 0.1753
Diffusion Epoch 53/60, Total Loss: 0.1110, Diff Loss: 0.0759, AR Loss: 0.1759
Diffusion Epoch 54/60, Total Loss: 0.1098, Diff Loss: 0.0748, AR Loss: 0.1748
Diffusion Epoch 55/60, Total Loss: 0.1102, Diff Loss: 0.0750, AR Loss: 0.1757
Diffusion Epoch 56/60, Total Loss: 0.1099, Diff Loss: 0.0748, AR Loss: 0.1755
Diffusion Epoch 57/60, Total Loss: 0.1093, Diff Loss: 0.0743, AR Loss: 0.1750
Diffusion Epoch 58/60, Total Loss: 0.1092, Diff Loss: 0.0742, AR Loss: 0.1749
Diffusion Epoch 59/60, Total Loss: 0.1086, Diff Loss: 0.0736, AR Loss: 0.1753
Diffusion Epoch 60/60, Total Loss: 0.1088, Diff Loss: 0.0737, AR Loss: 0.1754

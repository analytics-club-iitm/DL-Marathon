import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2

# progress bar without tqdm :P
def progress_bar(progress=0, status="", bar_len=20):
    status = status.ljust(30)
    block = int(round(bar_len * progress))
    text = "\rProgress: [{}] {}% {}".format(
        "\x1b[32m" + "#" * block + "\033[0m" + "-" * (bar_len - block), round(progress * 100, 2), status
    )
    print(text, end="")
    if progress == 1:
        print("\n")

# Discriminator
class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        leaky_relu = nn.LeakyReLU(0.2)
        dropout = nn.Dropout2d(0.2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            leaky_relu,
            dropout,
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            leaky_relu,
            dropout,
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            leaky_relu,
            dropout,
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            leaky_relu,
            dropout,
            nn.BatchNorm2d(128),
        )
        
        self.mlp = nn.Linear(128*2*2, 1)
        
    def forward(self, x):
        
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return torch.sigmoid(x)

# Generator
class Generator(nn.Module):
    
    def __init__(self):
    
        super(Generator, self).__init__()
        relu = nn.ReLU()
        self.mlp = nn.Linear(100, 8*8*128, bias = False)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(128),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            relu,
            nn.BatchNorm2d(128),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            relu,
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
        )
        
    def forward(self, x):
        x = self.mlp(x)
        x = x.reshape(x.shape[0], 128, 8, 8)
        x = self.conv(x)
        return torch.tanh(x)
    
if __name__ == "__main__":
    
    # variables
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_dir = "./data"
    img_size = 32
    latent_dim = 100
    num_epochs = 20
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # img transform and dataset
    img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5]) # mnist images are gray scale hence singe channel
    ])
    img_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=img_transform)
    img_loader = DataLoader(img_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # define model, optimizer, criterion (BCELoss)
    D = Discriminator().to(device)
    G = Generator().to(device)

    D_optim = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    G_optim = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    for epoch in range(1, num_epochs+1):
        D_loss_cntr, G_loss_cntr = [], []
        
        for i, (imgs, _) in enumerate(img_loader):
            
            imgs = imgs.to(device)        
            real_labels = torch.ones((imgs.shape[0], 1), device=device)
            fake_labels = torch.zeros((imgs.shape[0], 1), device=device)
            latent_vec = torch.randn((imgs.shape[0], latent_dim), device=device)

            # Train D
            real_logits = D(imgs)
            real_loss = criterion(real_logits, real_labels)
            
            fake_imgs = G(latent_vec)
            fake_logits = D(fake_imgs.detach())
            fake_loss = criterion(fake_logits, fake_labels)
            
            D_loss = real_loss + fake_loss
            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()
            
            # Train G
            G_loss = criterion(D(fake_imgs), real_labels)
            
            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()
            
            D_loss_cntr.append(D_loss.item())
            G_loss_cntr.append(G_loss.item())
            
            progress_bar(i/len(img_loader), status=f"Epoch: {epoch}, D Loss: {round(np.mean(D_loss_cntr), 3)}, G Loss: {round(np.mean(G_loss_cntr), 3)}")
            
            if i%100 == 0:
                frame = utils.make_grid(fake_imgs.detach().cpu(), nrow=8, normalize=True, range=(-1, 1))
                cv2.imshow("G images", frame.permute(1, 2, 0).numpy())
                if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
                    exit()
        
        progress_bar(1, status=f"Epoch: {epoch}, D Loss: {round(np.mean(D_loss_cntr), 3)}, G Loss: {round(np.mean(G_loss_cntr), 3)}")
        torch.save(G.state_dict(), os.path.join(output_dir, f'G-{epoch}.pth'))
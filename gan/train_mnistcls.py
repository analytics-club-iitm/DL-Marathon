import torch.nn as nn
import torch
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, utils
import torch.optim as optim
import numpy as np

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

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(6*6*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 6*6*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    
    # variables
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_size = 32
    batch_size = 64
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # img transform and train, test datasets
    img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # mnist images are gray scale hence singe channel
    ])
    trainset = datasets.MNIST('./data', download=True, train=True, transform=img_transform)
    valset = datasets.MNIST('./data', download=True, train=False, transform=img_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = Classifier().to(device)
    cls_optim = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):
        
        model.train()
        acc_cntr, loss_cntr = [], []
        for indx, (img, target) in enumerate(train_loader):
            
            img = img.to(device)
            target = target.to(device)
            out = model(img)

            cls_optim.zero_grad()
            loss = criterion(out, target)
            loss.backward()
            cls_optim.step()

            pred = out.max(1)[1]
            correct = pred.eq(target.long().data).cpu().sum()
            acc_cntr.append(correct.item()/img.shape[0])
            loss_cntr.append(loss.item())
            progress_bar(progress=indx/len(train_loader), status=f"Epoch: {epoch}, Train loss: {round(np.mean(loss_cntr),3)}, acc: {round(np.mean(acc_cntr),3)}")
        progress_bar(progress=1, status=f"Epoch: {epoch}, Train loss: {round(np.mean(loss_cntr),3)}, acc: {round(np.mean(acc_cntr),3)}")
        
        # testing
        model.eval()
        acc_cntr = []
        for indx, (img, target) in enumerate(test_loader):
            
            img = img.to(device)
            target = target.to(device)
            with torch.no_grad():
                out = model(img)

            pred = out.max(1)[1]
            correct = pred.eq(target.long().data).cpu().sum()
            acc_cntr.append(correct.item()/img.shape[0])
            progress_bar(progress=indx/len(train_loader), status=f"Epoch: {epoch}, Test acc: {round(np.mean(acc_cntr),3)}")
        progress_bar(progress=1, status=f"Epoch: {epoch}, Test acc: {round(np.mean(acc_cntr),3)}")

        torch.save(model.state_dict(), os.path.join(output_dir, f'cls-{epoch}.pth'))
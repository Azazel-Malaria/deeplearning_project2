#lama@research25.saas.hku.hk version 3090
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from tqdm import tqdm
import torch.nn.functional as F
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

trainset = torchvision.datasets.CIFAR10(root='/home/lama/CIFAR_10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='/home/lama/CIFAR_10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class OptimizedAlexNet(nn.Module):
    def __init__(self, activation_fn='relu'):
        super(OptimizedAlexNet, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            self.activation,
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            self.activation,
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            self.activation,
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(384 * 4 * 4, 2048),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            self.activation,
            nn.Linear(1024, 10),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_model(model, trainloader, testloader, criterion, optimizer, scheduler, num_epochs=50):
    since = datetime.now()
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = running_corrects.double() / len(trainloader.dataset)
        
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.cpu().numpy())
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_running_loss / len(testloader.dataset)
        val_acc = val_running_corrects.double() / len(testloader.dataset)
        
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc.cpu().numpy())
        
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        scheduler.step(val_loss)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    time_elapsed = datetime.now() - since
    print(f'Training complete in {time_elapsed}')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    return model

def visualize_loss_landscape(model, criterion, data_loader, resolution=30, device='cuda:0'):
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    original_params = [p.data.clone() for p in params]
    direction1 = [torch.randn_like(p) for p in params]
    direction2 = [torch.randn_like(p) for p in params]

    for d in direction1:
        d /= torch.norm(d)
    for d in direction2:
        d /= torch.norm(d)
    
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in tqdm(range(resolution), desc="Calculating loss landscape"):
        for j in range(resolution):
            for p, o, d1, d2 in zip(params, original_params, direction1, direction2):
                p.data = o + X[i,j] * d1 + Y[i,j] * d2
            total_loss = 0
            count = 0
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                count += 1
                if count >= 50:
                    break
            
            Z[i,j] = total_loss / count
    for p, o in zip(params, original_params):
        p.data = o
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                          linewidth=0, antialiased=False, alpha=0.8)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    ax.set_title('Loss Landscape Visualization')
    

    plt.savefig('loss_landscape.png')
    plt.close()
    
    return X, Y, Z

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = OptimizedAlexNet().to(device)
    
    def label_smoothing_loss(outputs, labels, smoothing=0.1):
        num_classes = outputs.shape[-1]
        log_preds = F.log_softmax(outputs, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(smoothing / (num_classes - 1))
            true_dist.scatter_(1, labels.data.unsqueeze(1), 1 - smoothing)
        return torch.mean(-torch.sum(true_dist * log_preds, dim=-1))
    
    criterion = label_smoothing_loss
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    model = train_model(model, trainloader, testloader, criterion, optimizer, scheduler, num_epochs=50)
    
    torch.save(model.state_dict(), 'final_model.pth')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
    print("\nVisualizing loss landscape...")
    _, _, _ = visualize_loss_landscape(model, criterion, testloader, resolution=30, device=device)
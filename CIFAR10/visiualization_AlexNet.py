import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

class OptimizedAlexNet(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def load_test_data():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = CIFAR10(root='C:/Users/Azazel/Desktop/深度学习/PJ2/codes', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return testloader, classes

def visualize_filters(model, layer_num=0, save_path='conv_filters.png'):
    weights = model.features[layer_num].weight.data.cpu().numpy()
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    
    fig, axes = plt.subplots(8, 12, figsize=(15, 10))
    for i, ax in enumerate(axes.flat):
        if i < weights.shape[0]:
            filter_img = weights[i].transpose(1, 2, 0)
            ax.imshow(filter_img)
        ax.axis('off')
    plt.suptitle(f'Filters in Conv Layer {layer_num}')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved filters visualization to {save_path}")

def visualize_feature_maps(model, testloader, layer_indices=[0, 3, 6], save_prefix='feature_maps'):
    dataiter = iter(testloader)
    images, _ = next(dataiter)
    images = images.to(device)
    
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    handles = []
    for i, layer in enumerate(model.features):
        if i in layer_indices:
            handles.append(layer.register_forward_hook(get_activation(f'layer_{i}')))
    
    model.eval()
    with torch.no_grad():
        _ = model(images)
    
    for handle in handles:
        handle.remove()
    
    for layer_idx in layer_indices:
        act = activations[f'layer_{layer_idx}'].cpu()
        fig, axes = plt.subplots(4, 8, figsize=(15, 8))
        for i in range(4):
            for j in range(8):
                if j < act.shape[1]:
                    axes[i,j].imshow(act[i,j].numpy(), cmap='viridis')
                axes[i,j].axis('off')
        plt.suptitle(f'Feature maps at layer {layer_idx}')
        plt.savefig(f'{save_prefix}_layer{layer_idx}.png')
        plt.close()
        print(f"Saved feature maps for layer {layer_idx} to {save_prefix}_layer{layer_idx}.png")

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = OptimizedAlexNet().to(device)
    model.load_state_dict(torch.load('aug_AlexNet.pth', map_location=device))
    model.eval()
    testloader, classes = load_test_data()
    visualize_filters(model, layer_num=0)
    visualize_feature_maps(model, testloader)
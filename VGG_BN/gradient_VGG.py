import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display
import torch.multiprocessing as mp
from collections import defaultdict

# 添加这一行来设置多进程启动方法
mp.set_start_method('spawn', force=True)

from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader

def compute_gradient_variation(model, data_loader, device):
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    grad_norms = []
    grad_diffs = []
    data, target = next(iter(data_loader))
    data, target = data.to(device), target.to(device)
    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    ref_grad = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        current_grad = [p.grad for p in model.parameters() if p.grad is not None]
        grad_norm = torch.norm(torch.cat([g.flatten() for g in current_grad]))
        grad_norms.append(grad_norm.item())
        grad_diff = 0
        for g1, g2 in zip(ref_grad, current_grad):
            grad_diff += torch.norm(g1 - g2).item()
        grad_diffs.append(grad_diff)
        ref_grad = [g.clone() for g in current_grad]
    
    return {
        'grad_norms': grad_norms,
        'grad_diffs': grad_diffs
    }

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]
    batch_size = 128
    num_workers = 4
    epochs = 20
    os.makedirs('reports/figures/gradients', exist_ok=True)
    os.makedirs('reports/metrics', exist_ok=True)
    train_loader = get_cifar_loader(train=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = get_cifar_loader(train=False, batch_size=batch_size, num_workers=num_workers)
    all_metrics = {
        'with_bn': defaultdict(dict),
        'without_bn': defaultdict(dict)
    }


    def train_and_analyze(model, lr, model_type):
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        step_losses = []
        grad_metrics = {
            'norms': [],
            'diffs': []
        }
        
        epoch_bar = tqdm(range(epochs), desc=f'{model_type} (LR={lr})', position=0)
        for epoch in epoch_bar:
            model.train()
            batch_bar = tqdm(train_loader, desc='Batches', leave=False, position=1)
            ref_grad = None
            for data, target in batch_bar:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                current_grad = [p.grad.clone() for p in model.parameters() if p.grad is not None]
                grad_norm = torch.norm(torch.cat([g.flatten() for g in current_grad]))
                grad_metrics['norms'].append(grad_norm.item())
                
                if ref_grad is not None:
                    grad_diff = sum(torch.norm(g1 - g2).item() for g1, g2 in zip(ref_grad, current_grad))
                    grad_metrics['diffs'].append(grad_diff)
                
                ref_grad = current_grad
                optimizer.step()
                step_losses.append(loss.item())
                
                batch_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'grad_norm': f"{grad_norm.item():.4f}"
                })
        
        return step_losses, grad_metrics

    for lr in learning_rates:
        print(f"\n=============== Analyzing LR={lr} =============")
        model = VGG_A()
        losses, metrics = train_and_analyze(model, lr, 'Without BN')
        all_metrics['without_bn'][lr] = {
            'losses': losses,
            'grad_norms': metrics['norms'],
            'grad_diffs': metrics['diffs']
        }
        
        model_bn = VGG_A_BatchNorm()
        losses_bn, metrics_bn = train_and_analyze(model_bn, lr, 'With BN')
        all_metrics['with_bn'][lr] = {
            'losses': losses_bn,
            'grad_norms': metrics_bn['norms'],
            'grad_diffs': metrics_bn['diffs']
        }
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(metrics['norms'], 'b', label='Without BN')
        plt.plot(metrics_bn['norms'], 'r', label='With BN')
        plt.title(f'Gradient Norm (LR={lr})')
        plt.xlabel('Step')
        plt.ylabel('Gradient Norm')
        plt.legend()
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.plot(metrics['diffs'], 'b', label='Without BN')
        plt.plot(metrics_bn['diffs'], 'r', label='With BN')
        plt.title(f'Gradient Difference (LR={lr})')
        plt.xlabel('Step')
        plt.ylabel('Gradient Change')
        plt.legend()
        plt.grid()
        
        plt.tight_layout()
        plt.savefig(f'reports/figures/gradients/gradient_metrics_lr{lr}.png')
        plt.close()

    np.save('reports/metrics/training_metrics.npy', all_metrics)

    print("\n============ Gradient Metrics Summary ==============")
    for lr in learning_rates:
        print(f"\nLearning Rate: {lr}")
        avg_norm = np.mean(all_metrics['without_bn'][lr]['grad_norms'])
        max_diff = np.max(all_metrics['without_bn'][lr]['grad_diffs'])
        print(f"Without BN - Avg Grad Norm: {avg_norm:.4f}, Max Grad Diff: {max_diff:.4f}")
        avg_norm = np.mean(all_metrics['with_bn'][lr]['grad_norms'])
        max_diff = np.max(all_metrics['with_bn'][lr]['grad_diffs'])
        print(f"With BN    - Avg Grad Norm: {avg_norm:.4f}, Max Grad Diff: {max_diff:.4f}")

if __name__ == '__main__':
    main()
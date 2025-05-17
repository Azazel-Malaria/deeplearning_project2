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
def main():
    # 初始化设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]  # 不同步长
    batch_size = 128
    num_workers = 4
    epochs = 20
    
    # 准备目录
    os.makedirs('reports/figures/per_lr', exist_ok=True)
    os.makedirs('reports/models', exist_ok=True)
    
    # 准备数据
    train_loader = get_cifar_loader(train=True, batch_size=batch_size, num_workers=num_workers)
    
    # 存储所有结果
    all_results = {
        'with_bn': defaultdict(list),
        'without_bn': defaultdict(list)
    }

    # 训练函数（带tqdm进度条）
    def train_model(model, lr, desc):
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        step_losses = []
        
        # 使用双进度条
        epoch_bar = tqdm(range(epochs), desc=f'{desc} (Epochs)', position=0)
        for epoch in epoch_bar:
            model.train()
            batch_bar = tqdm(train_loader, desc=f'LR={lr} (Batches)', leave=False, position=1)
            for data, target in batch_bar:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                step_losses.append(loss.item())
                batch_bar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
            epoch_bar.set_postfix({'avg_loss': f'{np.mean(step_losses[-len(train_loader):]):.4f}'})
        return step_losses

    # 对每个学习率进行训练和可视化
    for lr in learning_rates:
        print(f"\n=== Training with LR={lr} ===")
        
        # 训练不带BN的模型
        model = VGG_A()
        losses = train_model(model, lr, 'Without BN')
        all_results['without_bn'][lr] = losses
        torch.save(model.state_dict(), f'reports/models/vgg_a_lr{lr}.pth')
        
        # 训练带BN的模型
        model_bn = VGG_A_BatchNorm()
        losses_bn = train_model(model_bn, lr, 'With BN')
        all_results['with_bn'][lr] = losses_bn
        torch.save(model_bn.state_dict(), f'reports/models/vgg_bn_lr{lr}.pth')
        
        # 立即绘制当前学习率的损失地形
        plt.figure(figsize=(10, 6))
        
        # 计算当前学习率的损失地形
        max_curve = [max(w, b) for w, b in zip(all_results['without_bn'][lr], all_results['with_bn'][lr])]
        min_curve = [min(w, b) for w, b in zip(all_results['without_bn'][lr], all_results['with_bn'][lr])]
        
        # 绘制不带BN的区域
        plt.fill_between(range(len(all_results['without_bn'][lr])), 
                        all_results['without_bn'][lr], min_curve,
                        color='blue', alpha=0.2, label='Without BN')
        
        # 绘制带BN的区域
        plt.fill_between(range(len(all_results['with_bn'][lr])), 
                        all_results['with_bn'][lr], min_curve,
                        color='red', alpha=0.2, label='With BN')
        
        plt.title(f'Loss Landscape (LR={lr})')
        plt.xlabel('Training Step')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'reports/figures/per_lr/loss_landscape_lr{lr}.png')
        plt.close()

    # 绘制最终汇总图
    plt.figure(figsize=(12, 6))
    colors = ['b', 'g', 'r', 'c']
    
    for i, lr in enumerate(learning_rates):
        # 计算每个学习率的损失地形边界
        max_curve = [max(w, b) for w, b in zip(all_results['without_bn'][lr], all_results['with_bn'][lr])]
        min_curve = [min(w, b) for w, b in zip(all_results['without_bn'][lr], all_results['with_bn'][lr])]
        
        # 绘制填充区域
        plt.fill_between(range(len(max_curve)), max_curve, min_curve,
                        color=colors[i], alpha=0.2, label=f'LR={lr}')
        
        # 绘制边界线
        plt.plot(max_curve, '--', color=colors[i], lw=0.5)
        plt.plot(min_curve, '--', color=colors[i], lw=0.5)
    
    plt.title('Loss Landscape Comparison (All Learning Rates)')
    plt.xlabel('Training Step')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/figures/final_loss_landscape.png')
    plt.close()

    print("\nTraining completed. Results saved in reports/figures/")

if __name__ == '__main__':
    main()
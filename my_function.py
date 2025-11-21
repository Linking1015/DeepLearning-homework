import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch.nn.init as init


def load_data(dataset_name, batch_size, resize=None):
    """下载指定数据集并加载到内存，包含标准化处理"""
    
    # 定义数据集及其标准化参数
    dataset_info = {
        'mnist': {
            'class': torchvision.datasets.MNIST,
            'mean': (0.1307,), # 单通道
            'std': (0.3081,)
        },
        'fashion_mnist': {
            'class': torchvision.datasets.FashionMNIST,
            'mean': (0.2860,), # 单通道
            'std': (0.3530,)
        },
        'cifar10': {
            'class': torchvision.datasets.CIFAR10,
            # 三通道，这是CIFAR-10的常用全局统计数据
            'mean': (0.4914, 0.4822, 0.4465), 
            'std': (0.2023, 0.1994, 0.2010)
        }
    }

    if dataset_name not in dataset_info:
        raise ValueError(f"不支持的数据集: {dataset_name}。支持的数据集: {list(dataset_info.keys())}")

    info = dataset_info[dataset_name]
    dataset_class = info['class']

    # 构建数据转换管道
    trans = []
    if resize:
        trans.append(transforms.Resize(resize))
        
    trans.append(transforms.ToTensor())
    # **关键改进：添加标准化**
    trans.append(transforms.Normalize(info['mean'], info['std'])) 
    
    trans = transforms.Compose(trans)

    # 加载训练集和测试集
    train_set = dataset_class(root="./data", train=True, transform=trans, download=True)
    test_set = dataset_class(root="./data", train=False, transform=trans, download=True)

    # 创建数据加载器
    return (data.DataLoader(train_set, batch_size=batch_size, shuffle=True),
            data.DataLoader(test_set, batch_size=batch_size, shuffle=False))


def load_data_mnist(batch_size, resize=None):
    return load_data('mnist', batch_size, resize)


def load_data_fashion_mnist(batch_size, resize=None):
    return load_data('fashion_mnist', batch_size, resize)


def load_data_cifar10(batch_size, resize=None):
    # CIFAR-10 是 32x32，不需要额外 resize
    return load_data('cifar10', batch_size, resize)


class TrainingLogger:
    def __init__(self, model_name=None):
        """
        初始化训练记录器。
        注意：不再在 __init__ 中处理文件系统路径创建。
        """
        self.history = {
            'loss': [],  # 训练损失
            'train_acc': [],  # 训练准确率
            'val_acc': [],  # 验证准确率
            'epochs': []  # 训练轮次
        }
        self.model_name = model_name

    def update(self, epoch, loss=None, train_acc=None, val_acc=None):
        """更新训练记录"""
        self.history['epochs'].append(epoch)

        if loss is not None:
            self.history['loss'].append(loss)
        if train_acc is not None:
            self.history['train_acc'].append(train_acc)
        if val_acc is not None:
            self.history['val_acc'].append(val_acc)

    def plot_training_curves(self, save_dir=None, figsize=(10, 6), save=True):
        """
        Plot training curves (Loss and Accuracy). 
        保存路径 (save_dir) 现在在调用时传入。
        """
        epochs = self.history['epochs']
    
        fig, ax1 = plt.subplots(figsize=figsize)
    
        # 左边 Y 轴 - Loss
        color = 'tab:blue'
        if self.history['loss']:
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss', color=color)
            loss_line = ax1.plot(epochs, self.history['loss'],
                                 color=color, linestyle='-', linewidth=2, label='Training Loss')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
    
        # 右边 Y 轴 - Accuracy
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Accuracy', color=color)
    
        lines = []
    
        # Training Accuracy
        if self.history['train_acc']:
            train_acc_line = ax2.plot(epochs, self.history['train_acc'],
                                      color='tab:orange', linestyle='-', linewidth=2,
                                      label='Training Accuracy')
            lines.append(train_acc_line[0])
    
        # Validation Accuracy
        if self.history['val_acc']:
            val_acc_line = ax2.plot(epochs, self.history['val_acc'],
                                    color='tab:green', linestyle='-', linewidth=2,
                                    label='Validation Accuracy')
            lines.append(val_acc_line[0])
    
        ax2.tick_params(axis='y', labelcolor=color)
    
        # 合并图例
        if self.history['loss']:
            lines = [loss_line[0]] + lines
    
        ax1.legend(lines, [line.get_label() for line in lines], loc='center right')
    
        plt.title(f'{self.model_name} Training Curves') # 添加模型名称到标题
        plt.tight_layout()
    
        # 保存图片
        if save and save_dir:
            # 确保保存目录存在
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_curves_{self.model_name}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to: {filepath}")
    
        plt.show()  # 显示图像
        plt.close(fig)

    def get_best_epoch(self, metric='val_acc', mode='max'):
        """
        获取最佳epoch

        参数:
            metric: 评估指标 ('val_acc', 'train_acc', 'loss')
            mode: 'max' 或 'min'，表示指标是越大越好还是越小越好

        返回:
            (best_epoch, best_value): 最佳epoch和对应的指标值
        """
        if not self.history[metric]:
            return None, None

        if mode == 'max':
            best_value = max(self.history[metric])
            best_idx = self.history[metric].index(best_value)
        else:  # mode == 'min'
            best_value = min(self.history[metric])
            best_idx = self.history[metric].index(best_value)

        best_epoch = self.history['epochs'][best_idx]

        return best_epoch, best_value

    def print_summary(self):
        """打印训练摘要"""
        if not self.history['epochs']:
            print("没有训练记录")
            return

        print(f"--- {self.model_name} 训练摘要 ---")
        print(f"总训练轮次: {len(self.history['epochs'])}")

        if self.history['loss']:
            print(f"最终训练损失: {self.history['loss'][-1]:.4f}")
        if self.history['train_acc']:
            print(f"最终训练准确率: {self.history['train_acc'][-1]:.4f}")
        if self.history['val_acc']:
            print(f"最终验证准确率: {self.history['val_acc'][-1]:.4f}")

        # 显示最佳验证准确率
        if self.history['val_acc']:
            best_epoch, best_acc = self.get_best_epoch('val_acc', 'max')
            print(f"最佳验证准确率: {best_acc:.4f} (第 {best_epoch} 轮)")


def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed) # 如果不需要CUDA可以注释
    # torch.cuda.manual_seed_all(seed) # 如果不需要CUDA可以注释
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def default_init_weights(m):
    """默认权重初始化函数：适用于 Conv2d 和 Linear 层"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)


def train_single_model(
    model, 
    model_name, 
    train_loader, 
    test_loader, 
    device, 
    num_epochs=10,
    optimizer_cls=optim.Adam,
    lr=0.001,
    criterion=nn.CrossEntropyLoss(),
    weight_init_func=default_init_weights,
    seed=None
):
    """
    使用指定参数训练单个模型。

    参数:
        model: PyTorch模型实例 (nn.Module)。
        model_name: 模型的名称（用于日志和打印）。
        train_loader: 训练数据加载器。
        test_loader: 测试/验证数据加载器。
        device: 训练设备 (e.g., 'cuda', 'cpu')。
        num_epochs: 训练轮次。
        optimizer_cls: 优化器类 (e.g., optim.Adam, optim.SGD)。
        lr: 学习率。
        criterion: 损失函数。
        weight_init_func: 权重初始化函数。
        seed: 可选的随机种子。
    """
    print(f"\n开始训练模型: {model_name} (种子: {seed})")
    
    # 1. 设置随机种子
    if seed is not None:
        set_seed(seed)
        
    # 2. 权重初始化
    model.apply(weight_init_func)
    model.to(device)

    # 3. 优化器
    optimizer = optimizer_cls(model.parameters(), lr=lr)
    
    # 4. 训练记录器
    logger = TrainingLogger(model_name=model_name)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = train_correct / train_total
        avg_loss = total_loss / train_total

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        
        # 记录最佳验证准确率
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # 更新记录
        logger.update(epoch + 1, avg_loss, train_acc, val_acc)

        print(f'{model_name} - Epoch [{epoch + 1}/{num_epochs}] - 损失: {avg_loss:.4f}, 训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}')

    return model, logger, best_val_acc
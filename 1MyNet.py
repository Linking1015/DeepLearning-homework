import my_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class MyModel(nn.Module):
    def __init__(self, input_shape, num_big_classes):
        super(MyModel, self).__init__()
        # input_shape for CIFAR-10 is (3, 32, 32)
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        # 32x32 -> 16x16 -> 8x8. The feature map size is 32 * 8 * 8 = 2048
        self.fc1 = nn.Linear(32 * (input_shape[1] // 4) * (input_shape[2] // 4), 128)
        self.fc2 = nn.Linear(128, num_big_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# 主程序
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 加载数据
# 由于 MyModel 针对 32x32 的 CIFAR-10 设计，我们使用 load_data_cifar10
train_loader, test_loader = my_function.load_data_cifar10(256)

# 定义3个不同的随机种子
seeds = [42, 123, 456]
model_names = ['MyModel_Run1', 'MyModel_Run2', 'MyModel_Run3']
input_shape = (3, 32, 32) # CIFAR-10

# 训练3个模型（使用不同的随机种子）
results = []
for i, (seed, model_name) in enumerate(zip(seeds, model_names)):
    # 每次循环创建新的模型实例
    model = MyModel(input_shape=input_shape, num_big_classes=10)
    
    # 使用新的可复用训练函数
    model_trained, logger, best_val_acc = my_function.train_single_model(
        model=model, 
        model_name=model_name, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        device=device,
        num_epochs=25,
        seed=seed 
    )
    
    results.append({
        'seed': seed,
        'name': model_name,
        'model': model_trained,
        'logger': logger,
        'best_val_acc': best_val_acc
    })
    print(f"{model_name} (种子 {seed}) 最佳验证准确率: {best_val_acc:.4f}\n")

# --- 结果分析 ---
print("=" * 15, "MyModel 性能对比", "=" * 15)
for result in results:
    result['logger'].print_summary()

# 计算统计信息
accs = [result['best_val_acc'] for result in results]
mean_acc = np.mean(accs)
std_acc = np.std(accs)
print(f"\n\n模型的准确率分别为:", [f'{a:.4f}' for a in accs])
print(f"平均验证准确率: {mean_acc:.4f}")
print(f"标准差: {std_acc:.4f}")
print(f"准确率范围: {min(accs):.4f} - {max(accs):.4f}")

# 找到表现最好的模型
best_result = max(results, key=lambda x: x['best_val_acc'])
print(f"\n最佳模型: {best_result['name']} (种子 {best_result['seed']})")
print(f"最佳验证准确率: {best_result['best_val_acc']:.4f}")

# 为最佳模型绘制训练曲线，并将其保存在 'result_best' 目录
print(f"\n为最佳模型 {best_result['name']} 绘制训练曲线并保存...")
best_result['logger'].plot_training_curves(save_dir='./result_best', save=True)

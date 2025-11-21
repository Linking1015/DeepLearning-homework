import my_function
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


# --- 适配 32x32 的 AlexNet 模型定义 ---
class AlexNet_32(nn.Module):
    """
    AlexNet 结构，修改以适配 32x32x3 的 CIFAR-10 输入。
    采用适用于小图像的结构变体。
    """
    def __init__(self, num_classes=10):
        super(AlexNet_32, self).__init__()
        self.features = nn.Sequential(
            # 32x32x3 -> 16x16x64
            nn.Conv2d(3, 64, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 16x16x64 -> 8x8x192
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 8x8x192 -> 8x8x384 (无池化)
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 8x8x384 -> 4x4x256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # 4x4x256 -> 2x2x256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 最终特征图大小是 256 滤波器，尺寸 2x2
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 4096), # 256 * 4 = 1024
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
# --- 主程序 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 加载 CIFAR-10 数据，不进行 resize (使用 32x32 原图)
train_loader, test_loader = my_function.load_data_cifar10(batch_size=128)

# 定义多次运行的参数
MODEL_NAME = "AlexNet_32x32"
SEEDS = [42, 123, 456]
NUM_EPOCHS = 25
LEARNING_RATE = 0.0001 # AlexNet 常用较小学习率

results = []
print("=" * 15, f"开始 {MODEL_NAME} 多次训练", "=" * 15)

# 训练3个 AlexNet 模型（使用不同的随机种子）
for i, seed in enumerate(SEEDS):
    # 每次循环创建新的模型实例
    model = AlexNet_32(num_classes=10)
    model_name_run = f"{MODEL_NAME}_Run{i+1}"
    
    # 使用可复用训练函数
    model_trained, logger, best_val_acc = my_function.train_single_model(
        model=model, 
        model_name=model_name_run, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        device=device,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        seed=seed 
    )
    
    results.append({
        'seed': seed,
        'name': model_name_run,
        'model': model_trained,
        'logger': logger,
        'best_val_acc': best_val_acc
    })
    print(f"{model_name_run} (种子 {seed}) 最佳验证准确率: {best_val_acc:.4f}\n")

# --- 结果分析 ---
print("\n" + "=" * 10, f"{MODEL_NAME} 性能对比", "=" * 10)
for result in results:
    result['logger'].print_summary()

# 计算统计信息
accs = [result['best_val_acc'] for result in results]
mean_acc = np.mean(accs)
std_acc = np.std(accs)

print(f"\n{MODEL_NAME} 的准确率分别为: {[f'{a:.4f}' for a in accs]}")
print(f"平均验证准确率: {mean_acc:.4f}")
print(f"标准差: {std_acc:.4f}")
print(f"准确率范围: {min(accs):.4f} - {max(accs):.4f}")

# 找到 AlexNet 中表现最好的模型
best_result = max(results, key=lambda x: x['best_val_acc'])
print(f"\n{MODEL_NAME} 最佳运行结果: {best_result['name']} (种子 {best_result['seed']})")
print(f"最佳验证准确率: {best_result['best_val_acc']:.4f}")

# 为最佳 AlexNet 模型绘制训练曲线
print(f"\n为最佳模型 {best_result['name']} 绘制训练曲线并保存...")
# 保存路径使用 ./result_AlexNet
best_result['logger'].plot_training_curves(save_dir='./result_best', save=True)
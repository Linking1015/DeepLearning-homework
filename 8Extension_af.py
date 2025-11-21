import my_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random


# --- 1. 模型定义 (修改：支持动态传入激活函数) ---
class MyModel(nn.Module):
    def __init__(self, input_shape, num_big_classes, act_func_class):
        super(MyModel, self).__init__()
        # 实例化传入的激活函数类
        self.act = act_func_class()
        
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32 * (input_shape[1] // 4) * (input_shape[2] // 4), num_big_classes)

    def forward(self, x):
        # 使用 self.act 代替原来的 F.relu
        x = self.pool1(self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# --- 2. 训练函数 (修改：接收激活函数类参数) ---
def train_single_model(model_name, train_loader, test_loader, device, num_epochs, fixed_seed, act_func_class):
    """
    使用固定的随机种子、指定的Epoch数和激活函数训练模型
    """
    print(f"\n开始训练模型: {model_name}")
    # 获取激活函数的名称用于打印
    act_name = act_func_class.__name__
    print(f"设置: Epochs={num_epochs}, Seed={fixed_seed}, Activation={act_name}")
    
    # 每次训练前强制设置相同的种子, 保证状态一致
    my_function.set_seed(fixed_seed)
    
    # 实例化模型时传入激活函数类
    model = MyModel(
        input_shape=(3, 32, 32), 
        num_big_classes=10, 
        act_func_class=act_func_class
    ).to(device)
    
    # 权重初始化
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            # 使用 kaiming_normal 初始化，根据激活函数调整 nonlinearity
            # Sigmoid 和 Tanh 通常推荐 xavier_normal，但为了控制变量这里保持一致或根据类型调整
            # 这里为了简单和统一，我们统一使用 kaiming_normal
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    # 固定使用 Adam 优化器以控制变量
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 创建记录器
    logger = my_function.TrainingLogger( 
        model_name=model_name
    )
    
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

        # 减少打印频率，每5轮打印一次，或者最后一轮打印
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            print(f'{model_name} - Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    return model, logger, best_val_acc


# --- 3. 主程序 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 加载数据
train_loader, test_loader = my_function.load_data_cifar10(batch_size=256)

# === 实验设置 ===
FIXED_SEED = 42             # 固定种子
epochs = 50

# 定义要测试的激活函数列表
activation_settings = [nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh]
results = []

# === 循环不同的激活函数设置 ===
for act_class in activation_settings:
    # 获取类名用于文件名 (例如 "ReLU", "Sigmoid")
    act_name = act_class.__name__
    model_name = f"MyModel_Act_{act_name}"
    
    model, logger, best_val_acc = train_single_model(
        model_name=model_name,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=epochs,
        fixed_seed=FIXED_SEED,
        act_func_class=act_class  # 传入当前的激活函数类
    )
    
    results.append({
        'activation': act_name,
        'name': model_name,
        'model': model,
        'logger': logger,
        'best_val_acc': best_val_acc
    })
    print(f"完成 {model_name} 训练。最佳验证准确率: {best_val_acc:.4f}\n")
    
    # 保存训练曲线
    logger.plot_training_curves(save_dir=f"./result/{model_name}", save=True)


# === 结果分析 ===
print("=" * 15, "Activation Function 实验对比", "=" * 15)
for result in results:
    print(f"激活函数: {result['activation']:<10} | 最佳验证准确率 = {result['best_val_acc']:.4f}")

# 找到表现最好的设置
best_result = max(results, key=lambda x: x['best_val_acc'])
print(f"\n最佳实验设置: Activation={best_result['activation']} (准确率: {best_result['best_val_acc']:.4f})")
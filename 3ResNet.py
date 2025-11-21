import my_function
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


# --- 模型创建函数：标准 ResNet-18 ---
def create_resnet18(num_classes=10):
    # 加载标准 ResNet-18 结构
    model = models.resnet18(weights=None) 
    
    # 获取最后一层输入特征数 (通常为 512)
    num_ftrs = model.fc.in_features
    
    # 替换最后一层，使其输出 10 类
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model


# ==========================================================
# 主程序
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- 关键修改：加载数据时 Resize 到 224x224 ---
# 标准 ResNet 期望输入为 224x224。
# 注意：这将显著增加显存占用和计算时间。
print("正在加载数据并进行 Resize (224x224)...")
train_loader, test_loader = my_function.load_data_cifar10(batch_size=128, resize=224)

# 定义多次运行的参数
MODEL_NAME = "ResNet18"
SEEDS = [42, 123, 456]
NUM_EPOCHS = 25
LEARNING_RATE = 0.001 

results = []
print("=" * 15, f"开始 {MODEL_NAME} 多次训练", "=" * 15)

# 训练3个 ResNet 模型（使用不同的随机种子）
for i, seed in enumerate(SEEDS):
    # 每次循环创建新的模型实例
    model = create_resnet18(num_classes=10)
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

# 找到 ResNet 中表现最好的模型
best_result = max(results, key=lambda x: x['best_val_acc'])
print(f"\n{MODEL_NAME} 最佳运行结果: {best_result['name']} (种子 {best_result['seed']})")
print(f"最佳验证准确率: {best_result['best_val_acc']:.4f}")

# 为最佳 ResNet 模型绘制训练曲线
print(f"\n为最佳模型 {best_result['name']} 绘制训练曲线并保存...")
# 保存路径使用 ./result_ResNet_Standard 以区分之前的版本
best_result['logger'].plot_training_curves(save_dir='./result_best', save=True)
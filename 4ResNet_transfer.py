import my_function
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

# --- 迁移学习模型构建函数 ---
def create_transfer_resnet18(num_classes=10):
    """
    创建用于迁移学习的 ResNet-18 模型。
    """
    # 1. 加载预训练模型 (使用 ImageNet 默认权重)
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    # 2. 冻结所有参数 (骨干网络)
    # for param in model.parameters():
    #     param.requires_grad = False
        
    # 3. 替换最后一层全连接层
    # 注意：新创建的层是随机初始化的，且默认 requires_grad=True
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# ==========================================================
# 主程序
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 加载数据 (必须 Resize 到 224x224 以匹配预训练模型)
print("正在加载数据并进行 Resize (224x224)...")
train_loader, test_loader = my_function.load_data_cifar10(batch_size=128, resize=224)

# 定义实验参数
MODEL_NAME = "ResNet18_Transfer"
NUM_RUNS = 3  # 运行3次
NUM_EPOCHS = 25 
LEARNING_RATE = 0.001 

results = []
print("=" * 15, f"开始 {MODEL_NAME} 多次微调实验", "=" * 15)

# 循环训练 3 次
for i in range(NUM_RUNS):
    # 1. 创建新模型 (骨干参数相同，但 FC 层是新随机生成的)
    model = create_transfer_resnet18(num_classes=10)
    run_name = f"{MODEL_NAME}_Run{i+1}"
    
    # 2. 训练模型
    model_trained, logger, best_val_acc = my_function.train_single_model(
        model=model, 
        model_name=run_name, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        device=device,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        weight_init_func=None,
        seed=None, 
    )
    
    results.append({
        'name': run_name,
        'model': model_trained,
        'logger': logger,
        'best_val_acc': best_val_acc
    })
    print(f"{run_name} 最佳验证准确率: {best_val_acc:.4f}\n")

# --- 结果分析 ---
print("\n" + "=" * 10, f"{MODEL_NAME} 性能对比", "=" * 10)

# 计算统计信息
accs = [result['best_val_acc'] for result in results]
mean_acc = np.mean(accs)
std_acc = np.std(accs)

print(f"\n{MODEL_NAME} 的准确率分别为: {[f'{a:.4f}' for a in accs]}")
print(f"平均验证准确率: {mean_acc:.4f}")
print(f"标准差: {std_acc:.4f}")
print(f"准确率范围: {min(accs):.4f} - {max(accs):.4f}")

# 找到表现最好的模型
best_result = max(results, key=lambda x: x['best_val_acc'])
print(f"\n{MODEL_NAME} 最佳运行结果: {best_result['name']}")
print(f"最佳验证准确率: {best_result['best_val_acc']:.4f}")

# 为最佳模型绘制训练曲线并保存
print(f"\n为最佳模型 {best_result['name']} 绘制训练曲线并保存...")
best_result['logger'].plot_training_curves(save_dir='./result_best', save=True)
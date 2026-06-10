
#解析计算+ANN

# # Part 1: Network Training
# 此项目的目的是训练一个“EC型电感的损耗预测模型”

# # Step 0: Import Packages
# In this demo, the neural network is synthesized using the PyTorch framework. Please install PyTorch according to the [official guidance](https://pytorch.org/get-started/locally/) , then import PyTorch and other dependent modules.

# In[14]:


# Import necessary packages

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import json
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import scoreatpercentile

from analytic_model import analytic_inductor_model


# # Step 1: Define Network Structure
# In this part, we define the structure of the feedforward neural network. Refer to the [PyTorch document](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html) for more details.

# In[15]:
def inverse_preprocess(x_scaled, input_scaler):
    """
    x_scaled: torch.Tensor, shape (batch, 9), 标准化+log后的输入
    input_scaler: sklearn StandardScaler 对象
    """
    # 1. 转为 numpy
    x_scaled_np = x_scaled.detach().cpu().numpy()

    # 2. 逆标准化
    x_log = input_scaler.inverse_transform(x_scaled_np)

    # 3. 逆 log10（只对原来取 log 的列）
    x_raw = np.empty_like(x_log)
    # columns = ['C','dc1','dc2','f','ht','i','lg1','Nx','Ny']
    for j in range(9):
        x_raw[:, j] = 10 ** x_log[:, j]- 1e-8

    return x_raw


class Net(nn.Module):
    def __init__(self, input_scaler, output_scalers):
        super(Net, self).__init__()
        '''
        Shared feature extraction layers (混合部分)
        Input: 9 features (C, dc1, dc2, f, ht, i, lg1, Nx, Ny)
        '''
        self.input_scaler = input_scaler  # ✅ 必须保存
        self.output_scalers = output_scalers  # ✅ 必须保存

        '''
        Branched output layers (分支部分)
        Each output has its own dedicated path
        '''
        self.branch_L = nn.Sequential(
            nn.Linear(9, 105),  # 电感值分支
            nn.ReLU(),
            nn.Linear(105, 106),  # 输出L
            nn.ReLU(),
            nn.Linear(106, 22),
            nn.ReLU(),
            nn.Linear(22, 1),

        )

        self.branch_Pw = nn.Sequential(
            nn.Linear(9, 114),  # 绕组损耗分支
            nn.ReLU(),
            nn.Linear(114, 69),
            nn.ReLU(),
            nn.Linear(69, 105),
            nn.ReLU(),
            nn.Linear(105, 46),
            nn.ReLU(),
            nn.Linear(46, 1),
            # 输出Pw

        )

        self.branch_Pc = nn.Sequential(
            nn.Linear(9, 49),  # 磁芯损耗分支
            nn.ReLU(),
            nn.Linear(49, 73),
            nn.ReLU(),
            nn.Linear(73, 123),
            nn.ReLU(),
            nn.Linear(123, 23),
            nn.ReLU(),
            nn.Linear(23, 1),
            # 输出Pc

        )

    def forward(self, x):
        L_delta = self.branch_L(x)
        Pw_delta = self.branch_Pw(x)
        Pc_delta = self.branch_Pc(x)

        x_raw = inverse_preprocess(x, self.input_scaler)
        # --- 解析模型 ---
        y_analytic = analytic_inductor_model(x_raw)  # (batch,3)
        # ---------------- 解析输出 log + 标准化 ----------------
        y_analytic_log = np.log10(y_analytic + 1e-8)  # 避免 log(0)
        y_analytic_scaled = np.column_stack([
            self.output_scalers[i].transform(y_analytic_log[:, i].reshape(-1, 1))
            for i in range(3)
        ])
        y_analytic_scaled = torch.from_numpy(y_analytic_scaled).to(x.device).float()
        # ---------------- 最终输出 = 解析 + Δ ----------------
        # Δ 矫正后
        L  = L_delta  + y_analytic_scaled[:, 0:1]
        Pw = Pw_delta + y_analytic_scaled[:, 1:2]
        Pc = Pc_delta + y_analytic_scaled[:, 2:3]

        # 保持原 return 形式
        return torch.cat([L, Pw, Pc], dim=1) # 合并输出为 [batch_size, 3]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# # Step 2: Load the Dataset
# In this part, we load and pre-process the dataset for the network training and testing. In this demo, a small dataset containing triangular waveforms measured with N87 ferrite material under different frequency, flux density, and duty ratio is used, which can be downloaded from the [MagNet GitHub](https://github.com/PrincetonUniversity/Magnet) repository under "tutorial".

# In[19]:



def get_dataset():
    """加载Excel数据并进行预处理（单位转换+标准化）"""
    # 1. 加载Excel文件
    file_path = r"E:\Backups\AI_inductor_paper\2_ANN_train\Dataset\L_all_train.csv"
    data = pd.read_csv(file_path)

    # 2. 提取输入输出列（Python从0开始计数）
    input_columns = [0, 1, 2, 3, 4, 5, 6, 8, 9]  # C,dc1,dc2,f,ht,i,lg1,Nx,Ny
    output_columns = [11, 12, 13]  # L(uH), Pw(W), Pc(W)

    inputs = data.iloc[:, input_columns].values.astype(np.float32)
    outputs = data.iloc[:, output_columns].values.astype(np.float32)

    # 3. 单位转换（Pw/Pc: mW → W）
    #outputs[:, 1:] = outputs[:, 1:] / 1000

    # 4. 对可能存在指数关系的变量取对数（保持原始代码思路）
    # 假设频率f需要对数变换（根据物理特性决定）
    inputs[:, 0] = np.log10(inputs[:, 0])
    inputs[:, 1] = np.log10(inputs[:, 1])
    inputs[:, 2] = np.log10(inputs[:, 2])
    inputs[:, 3] = np.log10(inputs[:, 3])
    inputs[:, 4] = np.log10(inputs[:, 4])  # f(kHz)取对数
    inputs[:, 5] = np.log10(inputs[:, 5])  # i取对数
    inputs[:, 6] = np.log10(inputs[:, 6])  # i取对数
    inputs[:, 7] = np.log10(inputs[:, 7])  # Nx取对数
    inputs[:, 8] = np.log10(inputs[:, 8])  # Ny取对数

    # 5. 标准化处理（每个特征/输出独立标准化）
    input_scaler = StandardScaler()
    scaled_inputs = input_scaler.fit_transform(inputs)

    # 6. 输出先 log10
    outputs_log = np.log10(outputs + 1e-8)  # 避免 log(0)，加个很小的数

    #标准化
    output_scalers = [StandardScaler() for _ in range(3)]
    scaled_outputs = np.column_stack([
        output_scalers[i].fit_transform(outputs_log[:, i].reshape(-1, 1))
        for i in range(3)
    ])

    # 6. 转换为PyTorch张量
    input_tensor = torch.FloatTensor(scaled_inputs)
    output_tensor = torch.FloatTensor(scaled_outputs)

    # 7. 数据统计验证
    print("\n=== 数据统计 ===")
    print(f"样本数量: {len(input_tensor)}")

    print("\n输入变量（标准化前）:")
    input_stats = pd.DataFrame(inputs,
                               columns=['C(mm)', 'dc1(mm)', 'dc2(mm)', 'log10(f/kHz)', 'ht(mm)', 'i(A)', 'lg1(mm)',
                                        'Nx', 'Ny'])
    print(input_stats.describe().loc[['mean', 'std', 'min', 'max']].round(2))

    print("\n输出变量（转换单位后）:")
    output_stats = pd.DataFrame(outputs, columns=['L(uH)', 'Pw(W)', 'Pc(W)'])
    print(output_stats.describe().loc[['mean', 'std', 'min', 'max']].round(4))

    # 8. 计算输出变量标准差（未标准化前的）
    output_std = outputs.std(axis=0)  # L, Pw, Pc 的 std，单位保持 uH 和 W

    return TensorDataset(input_tensor, output_tensor), input_scaler, output_scalers, output_std


# # Step 3: Training and Testing the Model
# In this part, we program the training and testing procedure of the network model. The loaded dataset is randomly split into training set, validation set, and test set. The output of the training is the state dictionary file (.sd) containing all the trained parameter values.

# In[20]:


def weighted_mse_loss(output, target, weights):
    return (weights * (output - target) ** 2).mean()


def main():
    # 固定随机种子
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 超参数
    NUM_EPOCH = 200
    BATCH_SIZE = 100
    DECAY_EPOCH = 50
    DECAY_RATIO = 0.5
    LR_INI = 0.00416  # 建议Sigmoid时略微降低初始学习率

    # 判断设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # ✅ 仅 CUDA 可用时启用 cuDNN 控制
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # 加载数据（已在get_dataset中使用MinMaxScaler归一化）
    dataset, input_scaler, output_scalers, output_std = get_dataset()

    """
    # 损失函数权重
    inv_std = 1.0 / output_std
    weights = inv_std / inv_std.sum() * 3
    loss_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"Loss weights based on output std: {loss_weights.cpu().numpy().round(4)}")
    """

    #手动设置函数权重
    loss_weights = torch.tensor([0.5,1,1.5], dtype=torch.float32, device=device)
    print(f"Loss weights based on output std: {loss_weights.cpu().numpy().round(4)}")

    # 数据集划分
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size]
    )


    """
    # === 提取并反归一化（包含对数还原） 将训练、验证、测试集另存为.csv文件并保存===
    def extract_and_inverse(subset):
        X = torch.stack([x for x, _ in subset]).numpy()
        y = torch.stack([y for _, y in subset]).numpy()

        # 反标准化
        X = input_scaler.inverse_transform(X)
        y = np.column_stack([
            output_scalers[i].inverse_transform(y[:, i].reshape(-1, 1)).flatten()
            for i in range(3)
        ])

        # 还原 log10（输入 + 输出）
        X = np.power(10, X)
        y = np.power(10, y) - 1e-8  # 对应之前加的 1e-8（为了避免 log(0)）

        # 输出值还原单位（mW）
        #y[:, 1:] *= 1000
        return X, y
        
    # === 获取各集的数据 ===
    X_train, y_train = extract_and_inverse(train_dataset)
    X_val, y_val = extract_and_inverse(valid_dataset)
    X_test, y_test = extract_and_inverse(test_dataset)

    # === 保存为 CSV ===
    input_names = ['C(mm)', 'dc1(mm)', 'dc2(mm)', 'f(kHz)', 'ht(mm)', 'i(A)', 'lg1(mm)', 'Nx', 'Ny']
    output_names = ['L(uH)', 'Pw(W)', 'Pc(W)']

    df_train = pd.DataFrame(np.column_stack([X_train, y_train]), columns=input_names + output_names)
    df_val = pd.DataFrame(np.column_stack([X_val, y_val]), columns=input_names + output_names)
    df_test = pd.DataFrame(np.column_stack([X_test, y_test]), columns=input_names + output_names)

    df_train.to_csv("train_data.csv", index=False)
    df_val.to_csv("val_data.csv", index=False)
    df_test.to_csv("test_data.csv", index=False)

    print("训练、验证、测试数据已成功导出为 train_data.csv、val_data.csv、test_data.csv")
    """
    kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == 'cuda' else {}
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    # 初始化模型
    net = Net(input_scaler=input_scaler, output_scalers=output_scalers).to(device)
    print("Number of parameters:", count_parameters(net))

    optimizer = optim.Adam(net.parameters(), lr=LR_INI)

    best_valid_loss = float('inf')
    train_losses, valid_losses = [], []

    for epoch in range(NUM_EPOCH):
        lr = LR_INI * (DECAY_RATIO ** (epoch // DECAY_EPOCH))
        optimizer.param_groups[0]['lr'] = lr

        net.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = weighted_mse_loss(outputs, targets, loss_weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        net.eval()
        valid_loss = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = weighted_mse_loss(outputs, targets, loss_weights)
                valid_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        valid_losses.append(valid_loss / len(valid_loader))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{NUM_EPOCH}] | '
                  f'Train Loss: {train_losses[-1]:.6f} | '
                  f'Valid Loss: {valid_losses[-1]:.6f} | '
                  f'LR: {lr:.6f}')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'model_state': net.state_dict(),
                'input_scaler': input_scaler,
                'output_scalers': output_scalers
            }, 'best_model_analy_optuna.pth')
    """
    # Loss曲线
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve ')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig('loss_curve_3_3.png')
    plt.show()
    """

    # 设置全局字体和大小，确保符合 IEEE 风格
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18  # 适合 IEEE 双栏论文的字号
    plt.rcParams['axes.linewidth'] = 0.5  # 坐标轴线宽
    plt.rcParams['savefig.dpi'] = 600  # 保存图像时的分辨率
    plt.rcParams['figure.dpi'] = 300  # 图像显示分辨率

    # 创建绘图
    plt.figure(figsize=(5, 4))  # 适应双栏宽度

    # 绘制训练损失和验证损失曲线
    plt.plot(train_losses, label='Train Loss', linewidth=3)
    plt.plot(valid_losses, label='Valid Loss', linewidth=3)

    # 设置坐标轴标签和标题
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    #plt.title('Loss Curve', fontsize=10)

    # 显示图例
    plt.legend(fontsize=18, loc='upper right', frameon=False)

    # 启用网格
    plt.grid(True, linewidth=0.3)

    # 调整布局，避免元素被裁切
    plt.tight_layout(pad=1)

    # 显示图像
    plt.show()

    # 保存高分辨率图像（可选）
    plt.savefig('loss_curve.svg', dpi=600, bbox_inches='tight')

    # 加载并测试
    checkpoint = torch.load('best_model_analy_optuna.pth', weights_only=False)
    net.load_state_dict(checkpoint['model_state'])
    # input_scaler = checkpoint['input_scaler']
    # output_scalers = checkpoint['output_scalers']
    net.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            y_true.append(targets.numpy())
            y_pred.append(net(inputs).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    for i in range(3):
        # 第一步：反标准化（得到 log 值）
        y_true_log = output_scalers[i].inverse_transform(y_true[:, i].reshape(-1, 1)).flatten()
        y_pred_log = output_scalers[i].inverse_transform(y_pred[:, i].reshape(-1, 1)).flatten()

        # 第二步：10^log 得到原始值（撤销 log10）
        y_true[:, i] = 10 ** y_true_log
        y_pred[:, i] = 10 ** y_pred_log

    """
    # 保存预测结果为CSV
    result_df = pd.DataFrame({
        'True_L_uH': y_true[:, 0],
        'Pred_L_uH': y_pred[:, 0],
        'True_Pw_W': y_true[:, 1],
        'Pred_Pw_W': y_pred[:, 1],
        'True_Pc_W': y_true[:, 2],
        'Pred_Pc_W': y_pred[:, 2],
    })

    result_df.to_csv('prediction_results_1.csv', index=False)
    print("\n预测结果已保存为 prediction_results_1.csv")
    """
    # 性能评估
    mse = np.mean((y_true - y_pred) ** 2)  # 均方差
    mae = np.mean(np.abs(y_true - y_pred))  # 平均绝对误差
    print(f"\nFinal Test Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")

    output_names = ['Inductance (uH)', 'Winding Loss (W)', 'Core Loss (W)']

    # 计算所有输出的相对误差矩阵
    rel_errors = np.abs(y_true - y_pred) / (y_true + 1e-12) * 100

    # 统计误差信息的字典
    summary = {}

    for i in range(3):
        error = rel_errors[:, i]
        print(f"\n{output_names[i]} - Relative Error:")
        print(
            f"Mean: {np.mean(error):.2f}% | Max: {np.max(error):.2f}% | 95th Percentile(%): {round(scoreatpercentile(error, 95), 4)}")
        print(f"{output_names[i]} - R2 Score: {r2_score(y_true[:, i], y_pred[:, i]):.4f}")

        summary[output_names[i]] = {
            "Average(%)": round(np.mean(error), 4),
            "RMS(%)": round(np.sqrt(np.mean(error ** 2)), 4),
            "95th Percentile(%)": round(scoreatpercentile(error, 95), 4),
            "Maximum(%)": round(np.max(error), 4)
        }

    """
    # 绘制误差分布直方图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, name in enumerate(output_names):
        ax = axes[i]
        ax.hist(rel_errors[:, i], bins=30, edgecolor='black', alpha=0.7)
        p95 = summary[name]["95th Percentile(%)"]
        ax.axvline(p95, color='red', linestyle='--', label=f'95th percentile: {p95:.2f}%')
        ax.set_title(f'Relative Error Distribution: {name}')
        ax.set_xlabel('Relative Error (%)')
        ax.set_ylabel('Frequency')
        ax.legend()

    # 可视化结果
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i in range(3):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], s=10, alpha=0.5)
        ax.plot([y_true[:, i].min(), y_true[:, i].max()],
                [y_true[:, i].min(), y_true[:, i].max()],
                'r--', linewidth=2)
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{output_names[i]}')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('scatter_results_1.png')
    plt.show()
    
    """



if __name__ == "__main__":
    main()


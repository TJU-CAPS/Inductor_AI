# 此函数的功能是取代FEA仿真，求取磁元件的感值和损耗。
# 输入是磁元件的尺寸、激励，输出是磁元件的电感值、绕组损耗、磁芯损耗

import torch
import numpy as np
from torch import nn

# === 模型定义 ===
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(9, 79),  # 扩大隐藏层维度以增强特征提取能力
            nn.ReLU(),

        )

        '''
        Branched output layers (分支部分)
        Each output has its own dedicated path
        '''
        self.branch_L = nn.Sequential(
            nn.Linear(79, 24),  # 电感值分支
            nn.ReLU(),
            nn.Linear(24, 20),  # 输出L
            nn.ReLU(),
            nn.Linear(20, 79),
            nn.ReLU(),
            nn.Linear(79, 1),

        )

        self.branch_Pw = nn.Sequential(
            nn.Linear(79, 88),  # 绕组损耗分支
            nn.ReLU(),
            nn.Linear(88, 75),
            nn.ReLU(),
            nn.Linear(75, 1),  # 输出Pw

        )

        self.branch_Pc = nn.Sequential(
            nn.Linear(79, 92),  # 磁芯损耗分支
            nn.ReLU(),
            nn.Linear(92, 62),
            nn.ReLU(),
            nn.Linear(62, 1),  # 输出Pc

        )

    def forward(self, x):
        shared_features = self.shared_layers(x)
        L = self.branch_L(shared_features)
        Pw = self.branch_Pw(shared_features)
        Pc = self.branch_Pc(shared_features)
        return torch.cat([L, Pw, Pc], dim=1)

# === 初始化模型和 scaler（加载一次即可）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('best_model_1.pth', map_location=device, weights_only=False)
net = Net().to(device)
net.load_state_dict(checkpoint['model_state'])
net.eval()
input_scaler = checkpoint['input_scaler']
output_scalers = checkpoint['output_scalers']

# === 预测函数 ===
def predict_inductor(inputs):
    """
    输入：
        inputs: 长度为 9 的 tuple 或 list，格式为：
        ( dc1(mm), dc2(mm), ht(mm), lg1(mm), Nx, Ny, c(mm), f(kHz), i(A) )
    输出：
        L (uH), Pw (W), Pc (W)
    """
    if len(inputs) != 9:
        raise ValueError("输入必须是包含 9 个元素的元组或列表：dc1, dc2, ht, lg1, Nx, Ny, c, f, i")

    dc1, dc2, ht, lg1, Nx, Ny, c, f, i = inputs

    # 构造输入数组并 log10 变换
    x = np.array([[c, dc1, dc2, f, ht, i, lg1, Nx, Ny]], dtype=np.float32)
    x_log = np.log10(x)
    x_scaled = input_scaler.transform(x_log)
    x_tensor = torch.FloatTensor(x_scaled).to(device)
    # === 模型推理 ===
    with torch.no_grad():
        y_scaled = net(x_tensor).cpu().numpy()

    y_physical = np.zeros_like(y_scaled)
    for j in range(3):
        y_log = output_scalers[j].inverse_transform(y_scaled[:, j].reshape(-1, 1)).flatten()
        y_physical[:, j] = 10 ** y_log

    return tuple(y_physical[0])

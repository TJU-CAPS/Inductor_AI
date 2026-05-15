# 此函数的功能是取代FEA仿真，求取磁元件的感值和损耗。
# 输入是磁元件的尺寸、激励，输出是磁元件的电感值、绕组损耗、磁芯损耗

import torch
import numpy as np
from torch import nn
from tqdm import tqdm

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
def predict_inductor(X_input, batch_size=20000):
    """
    支持分批 GPU 推理 + 进度条
        输入：
            X_input: ndarray, shape (N, 9), 顺序为：dc1, dc2, ht, lg1, Nx, Ny, c, f, i
        输出：
            Y_pred: ndarray, shape (N, 3), 返回 L (μH), Pw (W), Pc (W)
        """
    # === 变量顺序转换为模型所需（c, dc1, dc2, f, ht, i, lg1, Nx, Ny）
    N = X_input.shape[0]
    Y_pred_list = []

    for i in tqdm(range(0, N, batch_size), desc="GPU 推理中"):
        X_batch = X_input[i:i + batch_size]

        # === 变量顺序转换为模型输入顺序
        X_reorder = np.stack([
            X_batch[:, 6],  # c
            X_batch[:, 0],  # dc1
            X_batch[:, 1],  # dc2
            X_batch[:, 7],  # f
            X_batch[:, 2],  # ht
            X_batch[:, 8],  # i
            X_batch[:, 3],  # lg1
            X_batch[:, 4],  # Nx
            X_batch[:, 5],  # Ny
        ], axis=1)

        X_log = np.log10(X_reorder.astype(np.float32))
        X_scaled = input_scaler.transform(X_log)
        X_tensor = torch.from_numpy(X_scaled).float().to(device)

        with torch.no_grad():
            Y_scaled = net(X_tensor).cpu().numpy()

        Y_physical = np.zeros_like(Y_scaled)
        for j in range(3):
            y_log = output_scalers[j].inverse_transform(Y_scaled[:, j].reshape(-1, 1)).flatten()
            Y_physical[:, j] = 10 ** y_log

        # 损耗单位转换（mW → W）
        #Y_physical[:, 1:] /= 1000

        Y_pred_list.append(Y_physical)

    return np.vstack(Y_pred_list)  # shape: [N, 3]

import torch
import pandas as pd
import numpy as np
from torch import nn
from sklearn.metrics import r2_score
from scipy.stats import scoreatpercentile
from sklearn.preprocessing import StandardScaler
import torch.serialization
from torch.serialization import add_safe_globals
add_safe_globals([StandardScaler])
import matplotlib.pyplot as plt
import matplotlib as mpl
# 添加 sklearn 的 scaler 到白名单（仅当你信任模型文件）
torch.serialization.add_safe_globals([StandardScaler])


# === 设置设备（优先用 GPU） ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# === 加载模型 ===
checkpoint = torch.load('best_model_1_0.pth', map_location=device, weights_only=False)
net = Net()
net.load_state_dict(checkpoint['model_state'])
net.eval()

input_scaler = checkpoint['input_scaler']
output_scalers = checkpoint['output_scalers']

# === 加载新输入数据（含真实输出） ===
# 假设输入 CSV 格式与训练集一致
data = pd.read_csv('Test_200.csv')  # 你需要提供这个文件
#data = pd.read_excel('0_1000.xlsx')  # 你需要提供这个文件

#input_columns = ['C(mm)', 'dc1(mm)', 'dc2(mm)', 'f(kHz)', 'ht(mm)', 'i(A)', 'lg1(mm)', 'Nx', 'Ny']
#output_columns = ['L(uH)', 'Pw(W)', 'Pc(W)']

input_columns = [0, 1, 2, 3, 4, 5, 6, 8, 9]  # C,dc1,dc2,f,ht,i,lg1,Nx,Ny
output_columns = [11, 12, 13]  # L(uH), Pw(mW), Pc(mW)

X_raw = data.iloc[:, input_columns].values.astype(np.float32)
Y_true_raw = data.iloc[:, output_columns].values.astype(np.float32)

#Y_true_raw[:, 1:] = Y_true_raw[:, 1:] / 1000

# === log10变换（输入） ===
X_raw[:, 0:9] = np.log10(X_raw[:, 0:9])

# === 标准化 ===
X_scaled = input_scaler.transform(X_raw)
X_tensor = torch.FloatTensor(X_scaled)

# === 模型推理 ===
with torch.no_grad():
    Y_pred_scaled = net(X_tensor).numpy()

# === 反标准化 + 撤销 log10 ===
Y_pred = np.zeros_like(Y_pred_scaled)
for i in range(3):
    log_pred = output_scalers[i].inverse_transform(Y_pred_scaled[:, i].reshape(-1, 1)).flatten()
    Y_pred[:, i] = 10 ** log_pred

Y_true = Y_true_raw.copy()

# === 误差计算 ===
abs_error = np.abs(Y_true - Y_pred)
rel_error = abs_error / (Y_true + 1e-12) * 100
output_names = ['L(uH)', 'Pw(W)', 'Pc(W)']
summary = {}

p95_list = []  # 用于存储每个输出的95%误差

for i in range(3):
    p95_val = scoreatpercentile(rel_error[:, i], 95)
    p95_list.append(p95_val)

    print(f"\n{output_names[i]} 相对误差统计:")
    print(f"平均误差: {np.mean(rel_error[:, i]):.2f}%")
    print(f"RMS误差: {np.sqrt(np.mean(rel_error[:, i] ** 2)):.2f}%")
    print(f"最大误差: {np.max(rel_error[:, i]):.2f}%")
    print(f"95分位误差: {p95_val:.2f}%")
    print(f"R2得分: {r2_score(Y_true[:, i], Y_pred[:, i]):.4f}")

    summary[output_names[i]] = {
        "Mean(%)": round(np.mean(rel_error[:, i]), 2),
        "RMS(%)": round(np.sqrt(np.mean(rel_error[:, i] ** 2)), 2),
        "Max(%)": round(np.max(rel_error[:, i]), 2),
        "P95(%)": round(p95_val, 2),
        "R2": round(r2_score(Y_true[:, i], Y_pred[:, i]), 4),
    }

# === 新增部分：计算三个输出的95%误差的平均值 ===
mean_p95 = np.mean(p95_list)
print(f"\nMean 95th Percentile Error = {mean_p95:.3f}% "
      f"(L={p95_list[0]:.3f}%, Pw={p95_list[1]:.3f}%, Pc={p95_list[2]:.3f}%)")

summary["Mean P95(%)"] = round(mean_p95, 3)




for i in range(3):
    print(f"\n{output_names[i]} 相对误差统计:")
    print(f"平均误差: {np.mean(rel_error[:, i]):.2f}%")
    print(f"RMS误差: {np.sqrt(np.mean(rel_error[:, i] ** 2)):.2f}%")
    print(f"最大误差: {np.max(rel_error[:, i]):.2f}%")
    print(f"95分位误差: {scoreatpercentile(rel_error[:, i], 95):.2f}%")
    print(f"R2得分: {r2_score(Y_true[:, i], Y_pred[:, i]):.4f}")

    summary[output_names[i]] = {
        "Mean(%)": round(np.mean(rel_error[:, i]), 2),
        "RMS(%)": round(np.sqrt(np.mean(rel_error[:, i] ** 2)), 2),
        "Max(%)": round(np.max(rel_error[:, i]), 2),
        "P95(%)": round(scoreatpercentile(rel_error[:, i], 95), 2),
        "R2": round(r2_score(Y_true[:, i], Y_pred[:, i]), 4),
    }

# === 保存预测结果到CSV ===
result_df = data.copy()
result_df['Pred_L(uH)'] = Y_pred[:, 0]
result_df['Pred_Pw(W)'] = Y_pred[:, 1]
result_df['Pred_Pc(W)'] = Y_pred[:, 2]

result_df.to_csv('test_result_output_Table1.csv', index=False)
print("\n✅预测结果已保存为 test_result_output_Table1.csv")


# 性能评估
mse = np.mean((Y_true - Y_pred) ** 2)  # 均方差
mae = np.mean(np.abs(Y_true - Y_pred))  # 平均绝对误差
print(f"\nFinal Test Metrics:")
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")

output_names = ['Inductance (uH)', 'Winding Loss (W)', 'Core Loss (W)']

# 计算所有输出的相对误差矩阵
rel_errors = np.abs(Y_true - Y_pred) / (Y_true + 1e-12) * 100

# 统计误差信息的字典
summary = {}

for i in range(3):
    error = rel_errors[:, i]
    print(f"\n{output_names[i]} - Relative Error:")
    print(
        f"Mean: {np.mean(error):.2f}% | Max: {np.max(error):.2f}% | 95th Percentile(%): {round(scoreatpercentile(error, 95), 4)}")
    print(f"{output_names[i]} - R2 Score: {r2_score(Y_true[:, i], Y_pred[:, i]):.4f}")

    summary[output_names[i]] = {
        "Average(%)": round(np.mean(error), 4),
        "RMS(%)": round(np.sqrt(np.mean(error ** 2)), 4),
        "95th Percentile(%)": round(scoreatpercentile(error, 95), 4),
        "Maximum(%)": round(np.max(error), 4)
    }


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


# 可视化结果
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i in range(3):
    ax = axes[i]
    ax.scatter(Y_true[:, i], Y_pred[:, i], s=10, alpha=0.5)
    ax.plot([Y_true[:, i].min(), Y_true[:, i].max()],
            [Y_true[:, i].min(), Y_true[:, i].max()],
            'r--', linewidth=2)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{output_names[i]}')
    ax.grid(True)

plt.tight_layout()
plt.savefig('scatter_results_1.png')
plt.show()


# ==========================
# 绘图参数（论文高清版本）
# ==========================
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.linewidth'] = 0.4  # 坐标轴更细
mpl.rcParams['savefig.dpi'] = 600     # 导出分辨率高
mpl.rcParams['figure.dpi'] = 300      # 屏幕显示清晰
mpl.rcParams['pdf.fonttype'] = 42     # 确保矢量字体可嵌入PDF
mpl.rcParams['ps.fonttype'] = 42

# ==========================
# 汇总误差结果
# ==========================
summary = {}
for i in range(3):
    error = rel_errors[:, i]
    summary[output_names[i]] = {
        "Mean Error(%)": round(np.mean(error), 4),
        "RMS(%)": round(np.sqrt(np.mean(error ** 2)), 4),
        "95th Percentile(%)": round(scoreatpercentile(error, 95), 4),
        "Max Error(%)": round(np.max(error), 4)
    }

# ==========================
# 绘制误差直方图
# ==========================
fig, axes = plt.subplots(1, 3, figsize=(3.5, 2), dpi=600)
for i, name in enumerate(output_names):
    ax = axes[i]

    # 绘制直方图
    ax.hist(rel_errors[:, i], bins=30, edgecolor='black', alpha=0.7, label='Data')

    mean_err = summary[name]["Mean Error(%)"]
    max_err = summary[name]["Max Error(%)"]

    # 竖线标注
    ax.axvline(mean_err, color='blue', linestyle='-', linewidth=0.8, label=f'Mean err.: {mean_err:.2f}%')
    ax.axvline(max_err, color='red', linestyle='--', linewidth=0.8, label=f'Max.err.: {max_err:.2f}%')

    # 坐标与标题
    ax.set_title(f'{name}', fontsize=8, pad=1)
    ax.set_xlabel('Error (%)', fontsize=8)

    if i == 0:
        ax.set_ylabel('Count', fontsize=8)  # 仅最左图显示纵坐标
    else:
        ax.set_ylabel('')
        #ax.set_yticklabels([])  # 去除中右图纵坐标刻度

    ax.tick_params(axis='both', labelsize=6, width=0.4, length=1.5)
    ax.legend(fontsize=5, loc='upper right', frameon=False, handlelength=1.5)

# 布局紧凑
plt.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.3)

# 保存高清图像
plt.savefig('Error_Distribution_ECCE2026Asia.svg', dpi=600, bbox_inches='tight')
plt.show()



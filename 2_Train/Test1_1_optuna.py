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
from analytic_model import analytic_inductor_model
from torch.utils.data import TensorDataset, DataLoader
# 添加 sklearn 的 scaler 到白名单（仅当你信任模型文件）
torch.serialization.add_safe_globals([StandardScaler])
# === 加载新输入数据（含真实输出） ===
# 假设输入 CSV 格式与训练集一致
#data = pd.read_csv('Test_5000.csv')  # 你需要提供这个文件

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False          # 默认 False 即可，大多数情况下够用
plt.rcParams['mathtext.fontset'] = 'stix'    # 或 'cm'、'dejavusans'

# === 设置设备（优先用 GPU） ===
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

#print(f"Using device: {device}")

def inverse_preprocess(x_scaled, input_scaler):
    """
    x_scaled: torch.Tensor, shape (batch, 9), 标准化+log后的输入
    input_scaler: sklearn StandardScaler 对象
    """
    import numpy as np

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


# === 加载模型 ===
checkpoint = torch.load('best_model_analy_optuna.pth', map_location=device, weights_only=False)
# 加载数据（已在get_dataset中使用MinMaxScaler归一化）
input_scaler = checkpoint['input_scaler']
output_scalers = checkpoint['output_scalers']
net = Net(input_scaler, output_scalers)
net.load_state_dict(checkpoint['model_state'])
net.eval()



# === 加载新输入数据（含真实输出） ===
# 假设输入 CSV 格式与训练集一致
data = pd.read_csv('Test_5000.csv')  # 你需要提供这个文件
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

# === 保存完整结果到 CSV（输入 + 真实 + 预测 + 误差） ===
result_df = pd.DataFrame()

# ---------- 输入变量 ----------
result_df["dc1"] = data.iloc[:, 1].values
result_df["dc2"] = data.iloc[:, 2].values
result_df["ht"]  = data.iloc[:, 4].values
result_df["lg1"] = data.iloc[:, 6].values
result_df["Nx"]  = data.iloc[:, 8].values
result_df["Ny"]  = data.iloc[:, 9].values
result_df["c"]   = data.iloc[:, 0].values
result_df["f"]   = data.iloc[:, 3].values
result_df["i"]   = data.iloc[:, 5].values

# ---------- 真实输出 ----------
result_df["L_true(uH)"] = Y_true[:, 0]
result_df["Pw_true(W)"] = Y_true[:, 1]
result_df["Pc_true(W)"] = Y_true[:, 2]

# ---------- 预测输出 ----------
result_df["L_pred(uH)"] = Y_pred[:, 0]
result_df["Pw_pred(W)"] = Y_pred[:, 1]
result_df["Pc_pred(W)"] = Y_pred[:, 2]

# ---------- 误差 ----------
result_df["L_rel_err(%)"] = rel_error[:, 0]
result_df["Pw_rel_err(%)"] = rel_error[:, 1]
result_df["Pc_rel_err(%)"] = rel_error[:, 2]

# ---------- 保存 ----------
result_df.to_csv("test_result_output_M2_optuna.csv", index=False)
print("\n✅ 已保存完整测试结果到 test_result_output_M2_optuna.csv")


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
#plt.show()


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
fig, axes = plt.subplots(1, 3, figsize=(3.2, 1.3), dpi=300)
for i, name in enumerate(output_names):
    ax = axes[i]

    # ===== 直方图 =====
    ax.hist(rel_errors[:, i], bins=30, edgecolor='black', alpha=0.7)

    # ===== 取统计量 =====
    mean_err = summary[name]["Mean Error(%)"]
    max_err = summary[name]["Max Error(%)"]
    p95_err = summary[name]["95th Percentile(%)"]

    # ===== 竖线 =====
    ax.axvline(mean_err, color='blue', linestyle='-', linewidth=0.8,
               label=f'Mean: {mean_err:.2f}%')

    ax.axvline(p95_err, color='green', linestyle='-.', linewidth=0.8,
               label=f'95%: {p95_err:.2f}%')

    ax.axvline(max_err, color='red', linestyle='--', linewidth=0.8,
               label=f'Max: {max_err:.2f}%')

    # ===== 坐标 =====
    ax.set_xlabel('Error (%)', fontsize=7)

    '''
        if i == 0:
        ax.set_ylabel('Count', fontsize=7)
    else:
        ax.set_ylabel('')
    '''

    ax.tick_params(axis='both', labelsize=6, width=0.4, length=1.5)

    ax.legend(fontsize=5.5, loc='upper right', frameon=False, handlelength=1.3)

# ===== 布局 =====
plt.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.3)

# ===== 保存 =====
plt.savefig('Error_Distribution.png', dpi=600, bbox_inches='tight')
#lt.show()

#=============================================================
fig, axes = plt.subplots(1, 3, figsize=(3.2, 1.3), dpi=600)

for i, name in enumerate(output_names):
    ax = axes[i]

    # 散点图
    ax.scatter(Y_true[:, i], Y_pred[:, i],
               s=4, alpha=0.6, edgecolor='none')

    # 对角参考线
    min_val = min(Y_true[:, i].min(), Y_pred[:, i].min())
    max_val = max(Y_true[:, i].max(), Y_pred[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val],
            color='red', linestyle='--', linewidth=0.8)

    # 坐标轴标签 - 使用粗体数学符号 + 下标（推荐）
    ax.set_xlabel(r'$Y_{\mathrm{true}}$', fontsize=7)

    if i == 0:
        ax.set_ylabel(r'$Y_{\mathrm{pred}}$', fontsize=8)
    else:
        ax.set_ylabel('')

    #ax.set_ylabel(r'$Y_{\mathrm{pred}}$', fontsize=7)

    # 子图标题（output_names 中的名称）
    #ax.set_title(name, fontsize=8, pad=1.5)

    # 刻度设置
    ax.tick_params(axis='both', labelsize=7, width=0.4, length=1.5)

    # 网格（细网格，可选注释掉）
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # 强制 x/y 轴等比例
    ax.set_aspect('equal', adjustable='box')

# 紧凑布局
plt.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.3)

# 保存
plt.savefig('T_F.png', dpi=600, bbox_inches='tight')
plt.close()

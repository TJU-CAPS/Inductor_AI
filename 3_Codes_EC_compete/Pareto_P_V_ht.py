import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Data_Generate import inputs
from F_ANN_GPU import  predict_inductor
from Selector import prompt_filter_conditions,filter_by_conditions
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 ['SimHei'] 若用中文
matplotlib.rcParams['axes.unicode_minus'] = False

# === 1. 生成数据
print("⏳ 正在生成输入组合...")
X = np.array(inputs(n=None))  # shape: [N, 9]

# === 2. 批量预测
print("⏳ 正在进行GPU批量推理...")
Y = predict_inductor(X, batch_size=20000)    # shape: [N, 3], L, Pw, Pc

# === 3. 组合输入、输出变量
results = np.hstack([X, Y])  # shape: [N, 12]
columns = ['dc1', 'dc2', 'ht', 'lg1', 'Nx', 'Ny', 'c', 'f', 'i', 'L(uH)', 'Pw(W)', 'Pc(W)']

# 变量解析
dc1, dc2, ht = results[:, 0], results[:, 1], results[:, 2]
lg1, Nx, Ny, c = results[:, 3], results[:, 4], results[:, 5], results[:, 6]
f,i,L=results[:, 7], results[:, 8], results[:, 9]
Pw, Pc = results[:, 10], results[:, 11]
P = Pw + Pc

# === 计算体积 V 和投影面积 S（你之前定义的）
w = dc1 + 0.7 * 4 + Nx * c * 2 + dc2 * 2
l = dc1 + 0.7 * 2 + Nx * c * 2
h = Ny * c + 0.7 * 2 + ht * 2
V = w * l * h

# === 4. 显示总组合数据统计结果
    # 1)、输入变量统计
df = pd.DataFrame(results, columns=columns)
print("\n📌 各输入变量的实际取值（不重复）:")
for col in columns[:9]:  # 只统计前9个输入变量
    unique_vals = sorted(df[col].unique())
    preview = unique_vals[:10]
    ellipsis = " ..." if len(unique_vals) > 10 else ""
    print(f"{col:6s}: {preview}{ellipsis}（共 {len(unique_vals)} 个）")
    # 2)、输出变量统计
"""
#直方图
for out_col in ['L(uH)', 'Pw(W)', 'Pc(W)']:
    plt.figure(figsize=(6, 4))
    plt.hist(df[out_col], bins=100, color='steelblue', edgecolor='black')
    plt.xlabel(out_col)
    plt.ylabel("样本数")
    plt.title(f"{out_col} 分布直方图")
    plt.xscale("log")  # ✅ 设置横坐标为 log10 对数坐标
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()  # ✅ 不保存，直接展示
        #散点图
"""
"""
for out_col in ['L(uH)', 'Pw(W)', 'Pc(W)']:
    plt.figure(figsize=(8, 4))
    plt.scatter(V, df[out_col], s=2, alpha=0.4, color='steelblue')
    plt.xlabel("体积 V (mm³)")
    plt.ylabel(out_col)
    plt.title(f"{out_col} 分布散点图")
    plt.yscale("log")  # ✅ 若数据跨度大，可使用对数纵轴
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
"""
#图1 P-L-i
plt.figure(figsize=(6, 5))
plt.scatter(P, L, c=i, cmap='turbo', s=8, alpha=0.6)
plt.xlabel("总损耗 P (W)")
plt.ylabel("感值 L (uH)")
plt.title(f"图1")
plt.colorbar(label="i (A)")
plt.grid(True)
plt.tight_layout()
plt.show()
"""
# 图2 Pc-Pw-i
plt.figure(figsize=(6, 5))
plt.scatter(Pc, Pw, c=i, cmap='turbo', s=8, alpha=0.6)
plt.xlabel("磁芯损耗 Pc (W)")
plt.ylabel("绕组损耗 Pw (W)")
plt.title(f"图2")
plt.colorbar(label="i (A)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 图3 P-V-ht/dc1
plt.figure(figsize=(6, 5))
plt.scatter(P, V, c=8/3*dc2 / dc1, cmap='turbo', s=8, alpha=0.6)
plt.xlabel("损耗 P (W)")
plt.ylabel("体积 V (mm³)")
plt.title(f"图3")
plt.colorbar(label="8/3 * dc2/dc1")
plt.grid(True)
plt.tight_layout()
plt.show()
"""


# === 5.进一步筛选数据
    # 1). 用户输入筛选条件
condition_list = prompt_filter_conditions(columns)

    # 2). 根据条件筛选组合集
filtered_groups = filter_by_conditions(results, columns, condition_list)



# === 6.绘制Pareto
for idx, group in enumerate(filtered_groups):
    if len(group) == 0:
        continue

    # 变量解析
    dc1, dc2, ht = group[:, 0], group[:, 1], group[:, 2]
    lg1, Nx, Ny, c = group[:, 3], group[:, 4], group[:, 5], group[:, 6]
    Pw, Pc = group[:, 10], group[:, 11]
    P = Pw + Pc

    # === 计算体积 V 和投影面积 S（你之前定义的）
    w = dc1 + 0.7 * 4 + Nx * c * 2 + dc2 * 2
    l = dc1 + 0.7 * 2 + Nx * c * 2
    h = Ny * c + 0.7 * 2 + ht * 2
    V = w * l * h
    """
    # === 图一：P vs V，颜色 = ht/dc1
    plt.figure(figsize=(6, 5))
    plt.scatter(V, P, c=8/3*ht / dc1, cmap='turbo', s=8, alpha=0.6)
    plt.xlabel("体积 V (mm³)")
    plt.ylabel("总损耗 P (W)")
    plt.title(f"组 {idx+1} | 颜色表示 ht/dc1")
    plt.colorbar(label="8/3 * ht/dc1")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    """


    # === 图二：P vs V，颜色 = dc1/dc2
    plt.figure(figsize=(6, 5))
    plt.scatter(V, P, c=8/np.pi*dc2 / dc1, cmap='turbo', s=8, alpha=0.6)
    plt.xlabel("体积 V (mm³)")
    plt.ylabel("总损耗 P (W)")
    plt.title(f"组 {idx+1} | 颜色表示 dc1/dc2")
    plt.colorbar(label="8/3 * dc2/dc1")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === 提取 Pareto 前沿
    def pareto_front(V, P):
        data = np.vstack((V, P)).T
        is_pareto = np.ones(data.shape[0], dtype=bool)
        for i, point in enumerate(data):
            if is_pareto[i]:
                is_pareto[is_pareto] = np.any(data[is_pareto] < point, axis=1)
                is_pareto[i] = True
        return is_pareto

    pareto_mask = pareto_front(V, P)

    # === 计算 Pareto 前沿上的 8/3 * dc2 / dc1 并输出统计
    pareto_metric = (8 / 3) * dc2[pareto_mask] / dc1[pareto_mask]
    print(f"\n✅ 第 {idx+1} 组 Pareto 前沿上 8/3 * dc2/dc1 的数值统计：")
    print("最大值：", np.max(pareto_metric))
    print("最小值：", np.min(pareto_metric))
    print("平均值：", np.mean(pareto_metric))
    print("所有值：", pareto_metric)

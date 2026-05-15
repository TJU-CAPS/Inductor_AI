import numpy as np
import pandas as pd
from Data_Generate import inputs
from F_ANN_GPU import  predict_inductor
from Selector import prompt_filter_conditions,filter_by_conditions,interactive_selection_1
from Plot_data import plot_static_scatter
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 ['SimHei'] 若用中文
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib
matplotlib.use('TkAgg')  # 👈 强制使用交互式 GUI 后端（最常用、兼容性好）
import matplotlib.pyplot as plt
import datashader as ds
import datashader.transfer_functions as tf

import matplotlib.cm as cm
from matplotlib import colors as mcolors


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

#图1 P-L-i=============================================================

df1 = pd.DataFrame({
    'P': Pw + Pc,
    'L': L,
    'i': i
})
plot_static_scatter(df1, x='P', y='L', color='i',
                     title="图1: P-L-i (颜色 = 电流)",
                     filename="图1_P-L-i_带坐标.png",
                     xlabel="总损耗 P (W)", ylabel="感值 L (uH)"
                     )

#=======================================================================


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
    S = w * l
    Ae1 = np.pi*(dc1/2)*(dc1/2)
    Ae2 = dc1*dc2
    K = Ae2/Ae1

    # === 图二：P vs V，颜色 = Ae2/Ae1

    df2 = pd.DataFrame({
        'V': V,
        'P': Pw + Pc,
        'K': K
    })
    plot_static_scatter(df2, x='V', y='P', color='K',
                         title="图2: P-V-K (颜色 = 面积比)",
                         filename="图2_P-V-K_带坐标.png",
                         xlabel="体积 V (mm³)", ylabel="总损耗 P (W)"
                         )

final_pareto_data = []
all_filtered_data = []

for idx, group in enumerate(filtered_groups):
    if len(group) == 0:
        continue
    clicked_points, group_sel = interactive_selection_1(group, columns)

    # 收集数据
    if clicked_points.size > 0:
        final_pareto_data.append(clicked_points)
    if group_sel.size > 0:
        all_filtered_data.append(group_sel)

# ✅ 保存交互选中的点（点击过）
if final_pareto_data:
    all_points = np.vstack(final_pareto_data)
    df_out = pd.DataFrame(all_points, columns=columns)

    # 添加 V 和 S
    dc1, dc2, ht = df_out["dc1"], df_out["dc2"], df_out["ht"]
    Nx, Ny, c = df_out["Nx"], df_out["Ny"], df_out["c"]
    w = dc1 + 0.7 * 4 + Nx * c * 2 + dc2 * 2
    l = dc1 + 0.7 * 2 + Nx * c * 2
    h = Ny * c + 0.7 * 2 + ht * 2
    V = w * l * h
    S = w * l
    df_out["V(mm³)"] = V
    df_out["S(mm²)"] = S

    df_out.to_csv("Pareto_points_蓝色.csv", index=False)
    print("✅ 已保存交互选择的点：Pareto_points_蓝色.csv")

# ✅ 保存所有筛选出来的点（无论点没点）
if all_filtered_data:
    all_groups = np.vstack(all_filtered_data)
    df_all = pd.DataFrame(all_groups, columns=columns)

    dc1, dc2, ht = df_all["dc1"], df_all["dc2"], df_all["ht"]
    Nx, Ny, c = df_all["Nx"], df_all["Ny"], df_all["c"]
    w = dc1 + 0.7 * 4 + Nx * c * 2 + dc2 * 2
    l = dc1 + 0.7 * 2 + Nx * c * 2
    h = Ny * c + 0.7 * 2 + ht * 2
    V = w * l * h
    S = w * l
    df_all["V(mm³)"] = V
    df_all["S(mm²)"] = S

    df_all.to_csv("All_points_蓝色.csv", index=False)
    print("✅ 已保存所有筛选后的组合点：All_points_蓝色.csv")
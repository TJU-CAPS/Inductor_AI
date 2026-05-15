import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


file_path = r"G:\Projects\2-Maxwell+AI\Codes\AI_inductor_paper\3_Codes_EC_compete\Post_prossece\All_points.csv"
df = pd.read_csv(file_path)

# 数据读取
dc1_b = df.iloc[:, 0].values
dc2_b = df.iloc[:, 1].values
cNx_b = df.iloc[:, 2].values
V_b = df.iloc[:, 3].values
P_b = df.iloc[:, 4].values

dc1_p = df.iloc[:, 6].values
dc2_p = df.iloc[:, 7].values
cNx_p = df.iloc[:, 8].values
V_p = df.iloc[:, 9].values
P_p = df.iloc[:, 10].values

dc1_r = df.iloc[:, 12].values
dc2_r = df.iloc[:, 13].values
cNx_r = df.iloc[:, 14].values
V_r = df.iloc[:, 15].values
P_r = df.iloc[:, 16].values

dc1_g = df.iloc[:, 18].values
dc2_g = df.iloc[:, 19].values
cNx_g = df.iloc[:, 20].values
V_g = df.iloc[:, 21].values
P_g = df.iloc[:, 22].values

#参数计算
S1_b = np.pi*dc1_b**2/4
S2_b = dc1_b*dc2_b
K_b = S2_b/S1_b

S1_g = np.pi*dc1_g**2/4
S2_g = dc1_g*dc2_g
K_g = S2_g/S1_g

S1_r = np.pi*dc1_r**2/4
S2_r = dc1_r*dc2_r
K_r = S2_r/S1_r


# ===== 颜色范围 =====
vmin, vmax = 0.1, 0.9
norm = plt.Normalize(vmin=vmin, vmax=vmax)
cmap = 'turbo'

# ===== 创建图 =====
fig, axes = plt.subplots(1, 3, figsize=(8, 2.5), sharey=True)

# ================= (a) r =================
mask_r = np.isfinite(V_r) & np.isfinite(P_r) & np.isfinite(K_r)

sc = axes[0].scatter(
    V_r[mask_r],
    P_r[mask_r],
    c=K_r[mask_r],
    cmap=cmap,
    norm=norm,
    s=6
)

axes[0].set_xlabel('V')
axes[0].set_ylabel('P')

# ================= (b) g =================
mask_g = np.isfinite(V_g) & np.isfinite(P_g) & np.isfinite(K_g)

axes[1].scatter(
    V_g[mask_g],
    P_g[mask_g],
    c=K_g[mask_g],
    cmap=cmap,
    norm=norm,
    s=6
)

axes[1].set_xlabel('V')

# ================= (c) b =================
mask_b = np.isfinite(V_b) & np.isfinite(P_b) & np.isfinite(K_b)

axes[2].scatter(
    V_b[mask_b],
    P_b[mask_b],
    c=K_b[mask_b],
    cmap=cmap,
    norm=norm,
    s=6
)

axes[2].set_xlabel('V')

# ===== 统一横坐标范围 =====
for ax in axes:
    ax.set_xlim(0, 25000)
    ax.grid(alpha=0.3)

# ===== 去掉重复Y轴刻度 =====
axes[1].set_yticklabels([])
axes[2].set_yticklabels([])

# ===== (a)(b)(c) 放底部 =====
labels = ['(a)', '(b)', '(c)']
for i, ax in enumerate(axes):
    ax.text(
        0.5, -0.35,
        labels[i],
        transform=ax.transAxes,
        ha='center',
        va='center',
        fontsize=10
    )

# ===== colorbar（不挤图）=====
cbar = fig.colorbar(
    sc,
    ax=axes,
    fraction=0.025,
    pad=0.02
)
cbar.set_label('K')

# ===== 布局调整 =====
plt.subplots_adjust(bottom=0.28, right=0.85)

# ===== 保存 =====
plt.savefig('3scatter_K.png', dpi=300)
plt.show()
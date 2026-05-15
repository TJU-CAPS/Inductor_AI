import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 ['SimHei'] 若用中文
matplotlib.rcParams['axes.unicode_minus'] = False

matplotlib.use("TkAgg")   # 或者 "Qt5Agg"

# === 1. 读取CSV ===
file_path = r"G:\Projects\2-Maxwell+AI\Codes\AI_inductor_paper\3_Codes_EC_compete\All_points_蓝色.csv"
df = pd.read_csv(file_path)

# 假设前两列是 x 和 y
x = df.iloc[:, 12].values
y = df.iloc[:, 14].values

# === 2. 绘制散点图 ===

fig, ax = plt.subplots()
pts = ax.scatter(x, y, s=5, c="b", alpha=0.6)
ax.set_title("按住鼠标左键圈选，关闭窗口后保存结果")

selected_all = set()  # 保存所有被选中的点索引


# === 3. 定义选择回调函数 ===
def onselect(verts):
    global selected_all
    path = Path(verts)
    ind = np.nonzero(path.contains_points(np.c_[x, y]))[0]
    selected_all.update(ind)  # 累积选择
    print(f"本次选中了 {len(ind)} 个点，总共 {len(selected_all)} 个点")

    # 更新颜色：选中过的点标红
    pts.set_facecolors(["r" if i in selected_all else "b" for i in range(len(x))])
    fig.canvas.draw_idle()


# === 4. 启动交互 ===
lasso = LassoSelector(ax, onselect)
plt.show()

# === 5. 窗口关闭后，导出选中的点 ===
if selected_all:
    df.iloc[list(selected_all)].to_csv("蓝色pareto_points.csv", index=False)
    print(f"已保存 {len(selected_all)} 个点到 蓝色pareto_points.csv")
else:
    print("没有选择任何点，未保存文件。")

import pandas as pd
def plot_static_scatter(df, x, y, color, title, filename, xlabel, ylabel, max_points=200000):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    # 找出L最大和最小对应的行
    idx_min = df[y].idxmin()
    idx_max = df[y].idxmax()
    df_min = df.loc[[idx_min]]
    df_max = df.loc[[idx_max]]

    # 排除极值点后进行随机采样
    df_rest = df.drop(index=[idx_min, idx_max])
    n_sample = max(0, max_points - 2)
    df_sample = df_rest.sample(n=min(n_sample, len(df_rest)), random_state=42)

    # 合并为最终绘图数据
    df_plot = pd.concat([df_min, df_max, df_sample], ignore_index=True)

    # 绘图
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(df_plot[x], df_plot[y], c=df_plot[color], cmap='turbo', s=8, alpha=0.6)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(color)

    plt.tight_layout()
    plt.savefig(filename)

    plt.ion()  # 打开非阻塞模式
    plt.show()
    plt.pause(2.0)  # 显示2秒（可调），期间窗口可交互
    plt.ioff()  # 关闭交互模式（避免影响后续）

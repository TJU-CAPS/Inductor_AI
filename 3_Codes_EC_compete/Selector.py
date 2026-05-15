import numpy as np

def select_by_conditions(
    data, columns,
    condition_dicts,       # 多组筛选条件：list of dict
    tolerances=None,       # 容差设置：如 {"L(uH)": 0.05}
    verbose=True
):
    """
    支持多组条件筛选，每组返回一份满足条件的 np.ndarray
    每个 conditions 是 dict，如 {"L(uH)": 10, "i": [2.5, 3.5], "Nx": 4}

    返回：
        List[np.ndarray]，每组对应一个子集
    """
    col_index = {name: idx for idx, name in enumerate(columns)}
    results = []

    for cond in condition_dicts:
        mask = np.ones(len(data), dtype=bool)
        for key, val in cond.items():
            if key not in col_index:
                continue
            idx = col_index[key]
            if isinstance(val, (list, tuple)) and len(val) == 2:
                mask &= (data[:, idx] >= val[0]) & (data[:, idx] <= val[1])
            elif isinstance(val, (int, float)):
                if tolerances and key in tolerances:
                    tol = tolerances[key]
                    mask &= (data[:, idx] >= val * (1 - tol)) & (data[:, idx] <= val * (1 + tol))
                else:
                    mask &= (data[:, idx] == val)
        subset = data[mask]
        results.append(subset)
        if verbose:
            print(f"✅ 条件 {cond} 匹配组合数: {len(subset)}")

    return results

def prompt_user_conditions(columns):
    print("\n🎯 请按提示输入每组筛选条件。支持如下字段：")
    print(", ".join(columns))
    print("格式示例： L(uH)=10±5%，i=2.5–3.5, Nx=4")
    print("输入空行或 done/exit 完成当前输入。\n")

    condition_list = []

    while True:
        raw = input("请输入一组筛选条件（或按回车结束）: ").strip()
        if raw.lower() in ("exit", "done", ""):
            break

        try:
            cond = {}
            items = [x.strip() for x in raw.split(",")]
            for item in items:
                if "=" not in item:
                    continue
                key, val = item.split("=")
                key = key.strip()
                val = val.strip()
                if "–" in val or "-" in val:  # 范围输入
                    val = val.replace("–", "-")  # 支持中文破折号
                    vmin, vmax = [float(x) for x in val.split("-")]
                    cond[key] = [vmin, vmax]
                elif val.endswith("%"):  # 容差输入（自动忽略，统一放 tolerances）
                    base = float(val[:-1])
                    cond[key] = base
                else:
                    cond[key] = float(val)
            condition_list.append(cond)
        except Exception as e:
            print("❌ 输入格式有误，请重试:", e)

    return condition_list


# 交互式输入筛选条件
def prompt_filter_conditions(columns):
    """
    让用户交互式输入多组变量筛选条件。
    - 支持设定数值、范围（a-b）、\ 或空表示不限制
    - 支持误差容忍：如 L=10, L_tol=5 表示 ±5%；dc2=1.5, dc2_tol=10 同理
    """
    print("\n🧩 请输入每组筛选条件（支持 =数值, a-b 范围, 或 \\ 表示任意，不输入默认为任意）：")
    print(f"字段：{', '.join(columns)}")
    print("示例：dc1=5, Nx=3, Ny=3, L(uH)=10, L_tol=5, dc2=1.5, dc2_tol=10")
    print("输入空行或 'ok' 表示输入结束。")

    condition_list = []

    while True:
        raw = input("\n请输入一组条件: ").strip()
        if raw.lower() in ["", "ok"]:
            break

        cond = {}
        tol = {}

        try:
            items = [x.strip() for x in raw.split(",")]
            for item in items:
                if "=" not in item:
                    continue
                key, val = item.split("=")
                key = key.strip()
                val = val.strip()

                if val == "\\" or val == "":
                    continue  # 不做限定

                if key in ["L_tol", "dc2_tol"]:
                    # 例如 L_tol=5 → tol["L(uH)"] = 0.05
                    tol_key = "L(uH)" if key == "L_tol" else "dc2"
                    tol[tol_key] = float(val) / 100.0
                    continue

                if "-" in val:
                    vmin, vmax = [float(x) for x in val.split("-")]
                    cond[key] = [vmin, vmax]
                else:
                    cond[key] = float(val)

            condition_list.append((cond, tol))

        except Exception as e:
            print(f"❌ 输入格式错误：{e}")
            continue

    return condition_list

# 筛选数据集
def filter_by_conditions(data, columns, condition_list):
    col_index = {name: i for i, name in enumerate(columns)}
    result_groups = []

    for cond, tol in condition_list:
        mask = np.ones(len(data), dtype=bool)
        for key, val in cond.items():
            if key not in col_index:
                continue
            idx = col_index[key]
            if isinstance(val, list) and len(val) == 2:
                mask &= (data[:, idx] >= val[0]) & (data[:, idx] <= val[1])
            else:
                if key in tol:
                    err = tol[key]
                    mask &= (data[:, idx] >= val * (1 - err)) & (data[:, idx] <= val * (1 + err))
                else:
                    mask &= (data[:, idx] == val)

        filtered = data[mask]
        result_groups.append(filtered)
        print(f"✅ 条件 {cond} 匹配样本数: {len(filtered)}")

    return result_groups


# 交互式筛选并可视化 P-V-Pw/P 和 P-V-S 图 ===
def interactive_selection(group, columns):
    import matplotlib.pyplot as plt

    dc1, dc2, ht = group[:, 0], group[:, 1], group[:, 2]
    lg1, Nx, Ny, c = group[:, 3], group[:, 4], group[:, 5], group[:, 6]
    f, i, L = group[:, 7], group[:, 8], group[:, 9]
    Pw, Pc = group[:, 10], group[:, 11]
    P = Pw + Pc
    w = dc1 + 0.7 * 4 + Nx * c * 2 + dc2 * 2
    l = dc1 + 0.7 * 2 + Nx * c * 2
    h = Ny * c + 0.7 * 2 + ht * 2
    V = w * l * h
    S = w * l
    Ae1 = np.pi * (dc1 / 2) * (dc1 / 2)
    Ae2 = dc1 * dc2
    K = Ae2 / Ae1

    # 让用户输入筛选条件
    while True:
        print("\n📌 请输入进一步筛选的 P 和 V 范围（输入格式：min max），或输入 ok 退出：")
        cmd = input("请输入 P 的范围(W)，如 0 3： ").strip()
        if cmd.lower() == 'ok':
            break
        try:
            pmin, pmax = map(float, cmd.split())
            vmin, vmax = map(float, input("请输入 V 范围（mm³），如 1000 3000: ").split())
        except:
            print("❌ 输入格式错误，请重新输入")
            continue

        # 筛选点
        mask = (P >= pmin) & (P <= pmax) & (V >= vmin) & (V <= vmax)
        P_sel, V_sel, Pw_sel, S_sel = P[mask], V[mask], Pw[mask], S[mask]
        group_sel = group[mask]

        # === 图A: P-V-S
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        scatter2 = ax2.scatter(V_sel, P_sel, c=S_sel, cmap='turbo', s=10, alpha=0.6)
        ax2.set_xlabel("体积 V (mm³)")
        ax2.set_ylabel("总损耗 P (W)")
        ax2.set_title("图B: P-V-S")
        plt.colorbar(scatter2, label="投影面积 S (mm²)")
        ax2.grid(True)

        # 图B: P-V-K
        ratio = Pw_sel / P_sel
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        scatter1 = ax1.scatter(V_sel, P_sel, c=K, cmap='turbo', s=10, alpha=0.6)
        ax1.set_xlabel("体积 V (mm³)")
        ax1.set_ylabel("总损耗 P (W)")
        ax1.set_title("图A: P-V-K")
        plt.colorbar(scatter1, label="边柱面积 / 中柱面积")
        ax1.grid(True)

        # 记录用户点击的点
        clicked = []

        def on_click(event):
            for ax in [ax1, ax2]:
                if event.inaxes == ax:
                    x_click, y_click = event.xdata, event.ydata
                    dist = np.sqrt((V_sel - x_click) ** 2 + (P_sel - y_click) ** 2)
                    idx = np.argmin(dist)
                    print("\n👉 你选中了一个点，对应变量如下：")
                    for name, val in zip(columns, group_sel[idx]):
                        print(f"{name:6s}: {val:.4f}")
                    clicked.append(group_sel[idx])
                    break

        cid1 = fig1.canvas.mpl_connect('button_press_event', on_click)
        cid2 = fig2.canvas.mpl_connect('button_press_event', on_click)

        plt.show()

        print("👉 请点击图中点查看变量，输入 ok 回车可退出本轮交互...")

        # 交互等待输入退出
        while True:
            cmd2 = input("输入 ok 结束当前图交互（或按图中任意点查看变量）：").strip()
            if cmd2.lower() == 'ok':
                plt.close(fig1)
                plt.close(fig2)
                break

        return np.array(clicked)

"""
def interactive_selection_1(group, columns):
    import matplotlib.pyplot as plt
    import numpy as np

    dc1, dc2, ht = group[:, 0], group[:, 1], group[:, 2]
    lg1, Nx, Ny, c = group[:, 3], group[:, 4], group[:, 5], group[:, 6]
    f, i, L = group[:, 7], group[:, 8], group[:, 9]
    Pw, Pc = group[:, 10], group[:, 11]
    P = Pw + Pc
    w = dc1 + 0.7 * 4 + Nx * c * 2 + dc2 * 2
    l = dc1 + 0.7 * 2 + Nx * c * 2
    h = Ny * c + 0.7 * 2 + ht * 2
    V = w * l * h
    S = w * l
    Ae1 = np.pi * (dc1 / 2) ** 2
    Ae2 = dc1 * dc2
    K = Ae2 / Ae1

    while True:
        print("\n📌 请输入进一步筛选的 P 和 V 范围（输入格式：min max），或输入 ok 退出：")
        cmd = input("请输入 P 的范围(W)，如 0 3： ").strip()
        if cmd.lower() == 'ok':
            return np.array([])  # 用户放弃选择

        try:
            pmin, pmax = map(float, cmd.split())
            vmin, vmax = map(float, input("请输入 V 范围（mm³），如 1000 3000: ").split())
        except:
            print("❌ 输入格式错误，请重新输入")
            continue

        # 筛选数据
        mask = (P >= pmin) & (P <= pmax) & (V >= vmin) & (V <= vmax)
        if not np.any(mask):
            print("⚠️ 无符合条件的数据点")
            continue

        P_sel, V_sel = P[mask], V[mask]
        Pw_sel, S_sel, K_sel = Pw[mask], S[mask], K[mask]
        group_sel = group[mask]

        clicked = []

        def on_click(event):
            if event.inaxes:
                x_click, y_click = event.xdata, event.ydata
                dist = np.sqrt((V_sel - x_click)**2 + (P_sel - y_click)**2)
                idx = np.argmin(dist)
                print("\n👉 你选中了一个点，对应变量如下：")
                for name, val in zip(columns, group_sel[idx]):
                    print(f"{name:6s}: {val:.4f}")
                clicked.append(group_sel[idx])


        # 图A：P-V-S（颜色 = 投影面积）
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sc2 = ax2.scatter(V_sel, P_sel, c=K_sel, cmap='turbo', s=10, alpha=0.6)
        ax2.set_xlabel("体积 V (mm³)")
        ax2.set_ylabel("总损耗 P (W)")
        ax2.set_title("图A: P-V-K")
        plt.colorbar(sc2, label="边柱面积 / 中柱面积", ax=ax2)
        ax2.grid(True)
        fig2.canvas.mpl_connect('button_press_event', on_click)

        print("👉 请点击图中点查看变量，关闭图窗后返回。")

        plt.show()  # 阻塞式：直到图窗被关闭

        return np.array(clicked), group_sel
"""

def interactive_selection_2(group, columns):
    import matplotlib
    matplotlib.use("TkAgg")  # 独立窗口，支持交互
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.widgets import LassoSelector
    from matplotlib.path import Path

    # ==== 数据预处理 ====
    dc1, dc2, ht = group[:, 0], group[:, 1], group[:, 2]
    lg1, Nx, Ny, c = group[:, 3], group[:, 4], group[:, 5], group[:, 6]
    f, i, L = group[:, 7], group[:, 8], group[:, 9]
    Pw, Pc = group[:, 10], group[:, 11]
    P = Pw + Pc
    w = dc1 + 0.7 * 4 + Nx * c * 2 + dc2 * 2
    l = dc1 + 0.7 * 2 + Nx * c * 2
    h = Ny * c + 0.7 * 2 + ht * 2
    V = w * l * h
    S = w * l
    Ae1 = np.pi * (dc1 / 2) ** 2
    Ae2 = dc1 * dc2
    K = Ae2 / Ae1

    while True:
        print("\n📌 请输入进一步筛选的 P 和 V 范围（输入格式：min max），或输入 ok 退出：")
        cmd = input("请输入 P 的范围(W)，如 0 3： ").strip()
        if cmd.lower() == 'ok':
            return np.array([]), np.array([])  # 用户放弃选择

        try:
            pmin, pmax = map(float, cmd.split())
            vmin, vmax = map(float, input("请输入 V 范围（mm³），如 1000 3000: ").split())
        except:
            print("❌ 输入格式错误，请重新输入")
            continue

        # ==== 筛选数据 ====
        mask = (P >= pmin) & (P <= pmax) & (V >= vmin) & (V <= vmax)
        if not np.any(mask):
            print("⚠️ 无符合条件的数据点")
            continue

        P_sel, V_sel = P[mask], V[mask]
        K_sel = K[mask]
        group_sel = group[mask]

        selected_idx = set()  # 存储选中的索引

        # ==== 定义圈选回调 ====
        def onselect(verts):
            path = Path(verts)
            ind = np.nonzero(path.contains_points(np.c_[V_sel, P_sel]))[0]
            selected_idx.update(ind)
            print(f"👉 本次圈选 {len(ind)} 个点，总共 {len(selected_idx)} 个点")
            pts.set_facecolors(["r" if i in selected_idx else "b" for i in range(len(V_sel))])
            fig.canvas.draw_idle()

        # ==== 绘制散点图 ====
        fig, ax = plt.subplots(figsize=(6, 5))
        pts = ax.scatter(V_sel, P_sel, c=K_sel, cmap='turbo', s=10, alpha=0.6)
        ax.set_xlabel("体积 V (mm³)")
        ax.set_ylabel("总损耗 P (W)")
        ax.set_title("图A: P-V-K (鼠标左键圈选，关闭窗口后保存)")
        plt.colorbar(pts, label="边柱面积 / 中柱面积", ax=ax)
        ax.grid(True)

        lasso = LassoSelector(ax, onselect)
        plt.show()  # 阻塞，直到关闭窗口

        # ==== 返回结果 ====
        if selected_idx:
            clicked_points = group_sel[list(selected_idx)]
            return clicked_points, group_sel
        else:
            print("⚠️ 未选择任何点")
            return np.array([]), group_sel


def interactive_selection_1(group, columns, save_prefix="selection"):
    import matplotlib
    matplotlib.use("TkAgg")  # 确保独立窗口可交互
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.widgets import LassoSelector
    from matplotlib.path import Path

    # ==== 数据预处理 ====
    dc1, dc2, ht = group[:, 0], group[:, 1], group[:, 2]
    lg1, Nx, Ny, c = group[:, 3], group[:, 4], group[:, 5], group[:, 6]
    f, i, L = group[:, 7], group[:, 8], group[:, 9]
    Pw, Pc = group[:, 10], group[:, 11]
    P = Pw + Pc
    w = dc1 + 0.7 * 4 + Nx * c * 2 + dc2 * 2
    l = dc1 + 0.7 * 2 + Nx * c * 2
    h = Ny * c + 0.7 * 2 + ht * 2
    V = w * l * h
    S = w * l
    Ae1 = np.pi * (dc1 / 2) ** 2
    Ae2 = dc1 * dc2
    K = Ae2 / Ae1

    while True:
        print("\n📌 请输入进一步筛选的 P 和 V 范围（格式：min max），或输入 ok 退出：")
        cmd = input("P 范围(W)，如 0 3： ").strip()
        if cmd.lower() == 'ok':
            return np.array([]), np.array([])  # 用户放弃

        try:
            pmin, pmax = map(float, cmd.split())
            vmin, vmax = map(float, input("V 范围(mm³)，如 1000 3000: ").split())
        except:
            print("❌ 输入格式错误，请重新输入")
            continue

        # ==== 筛选数据 ====
        mask = (P >= pmin) & (P <= pmax) & (V >= vmin) & (V <= vmax)
        if not np.any(mask):
            print("⚠️ 无符合条件的数据点")
            continue

        P_sel, V_sel = P[mask], V[mask]
        K_sel = K[mask]
        group_sel = group[mask]

        selected_idx = set()  # 存储圈选索引

        # ==== 初始化颜色数组 ====
        colors = np.array(["b"] * len(V_sel))  # 所有点蓝色

        # ==== 定义圈选回调 ====
        def onselect(verts):
            path = Path(verts)
            ind = np.nonzero(path.contains_points(np.c_[V_sel, P_sel]))[0]
            selected_idx.update(ind)
            print(f"👉 本次圈选 {len(ind)} 个点，总共 {len(selected_idx)} 个点")

            # 更新颜色，高亮选中点
            for i in ind:
                colors[i] = "r"
            pts.set_color(colors)
            fig.canvas.draw_idle()

        # ==== 绘制散点图 ====
        fig, ax = plt.subplots(figsize=(6, 5))
        pts = ax.scatter(V_sel, P_sel, c=colors, s=20, alpha=0.8)
        ax.set_xlabel("体积 V (mm³)")
        ax.set_ylabel("总损耗 P (W)")
        ax.set_title("P-V-K (鼠标左键圈选，关闭窗口后保存)")
        plt.colorbar(plt.cm.ScalarMappable(cmap='turbo'), label="边柱面积/中柱面积", ax=ax)
        ax.grid(True)

        # 启动 LassoSelector
        lasso = LassoSelector(ax, onselect)
        plt.show()  # 阻塞，直到关闭窗口

        # ==== 返回结果并保存 CSV ====
        if selected_idx:
            clicked_points = group_sel[list(selected_idx)]

            # 保存 CSV
            df_out = pd.DataFrame(clicked_points, columns=columns)
            # 添加 V 和 S
            dc1_out, dc2_out, ht_out = df_out["dc1"], df_out["dc2"], df_out["ht"]
            Nx_out, Ny_out, c_out = df_out["Nx"], df_out["Ny"], df_out["c"]
            w_out = dc1_out + 0.7 * 4 + Nx_out * c_out * 2 + dc2_out * 2
            l_out = dc1_out + 0.7 * 2 + Nx_out * c_out * 2
            h_out = Ny_out * c_out + 0.7 * 2 + ht_out * 2
            df_out["V(mm³)"] = w_out * l_out * h_out
            df_out["S(mm²)"] = w_out * l_out

            filename = f"{save_prefix}_clicked_points_蓝色.csv"
            df_out.to_csv(filename, index=False)
            print(f"✅ 已保存交互选择的点到 {filename}")

            return clicked_points, group_sel
        else:
            print("⚠️ 未选择任何点")
            return np.array([]), group_sel



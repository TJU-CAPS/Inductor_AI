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
    - 支持 L(uH) 误差容忍：L=10, tol=5 表示 ±5%
    """
    print("\n🧩 请输入每组筛选条件（支持 =数值, a-b 范围, 或 \\ 表示任意，不输入默认为任意）：")
    print(f"字段：{', '.join(columns)}")
    print("示例：dc1=5, Nx=3, Ny=3, L(uH)=10, L_tol=5")
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

                if key == "L_tol":
                    tol["L(uH)"] = float(val) / 100.0
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
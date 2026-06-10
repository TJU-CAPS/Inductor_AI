import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import random
import os
import cupy as cp

# ------------------------ 参数定义 ------------------------
Bmax = 0.3  # T，最大磁通密度
mu = 4 * math.pi * 1e-7  # H/m，真空磁导率

# 变量范围和步长
dc1_range = np.arange(5, 30.1, 0.6,dtype=np.float16)
Nx_range = np.arange(1, 10.1, 1,dtype=np.float16)
Ny_range = np.arange(1, 10.1, 2,dtype=np.float16)
c_range = np.arange(1, 5.1, 0.5,dtype=np.float16)  #与maxwell利兹线股数n设定关联
f_range = np.arange(100, 500.1, 100,dtype=np.float16)
dc2_step = np.float32(0.5)
ht_step = np.float32(1)
lg1_step = np.float32(0.4)
i_step = np.float32(8)

# 样本数量要求
sample_sizes = [3000]
output_dir = "1_Data_inputs"
os.makedirs(output_dir, exist_ok=True)


# ------------------------ 优化后的GPU筛选 ------------------------
def gpu_filter_combinations():
    valid_samples = []

    # 一级筛选（CPU）
    print("生成一级变量组合...")
    level1_combos = []
    for dc1 in tqdm(dc1_range):
        for Nx in Nx_range:
            for Ny in Ny_range:
                for c in c_range:
                    if 0.1 < Nx * c < 20 and 0.1 < Ny * c < 20:
                        for f in f_range:
                            level1_combos.append((np.float32(dc1), np.float32(Nx), np.float32(Ny), np.float32(c), np.float32(f)))
    print(f"通过一级筛选的组合数: {len(level1_combos)}")

    # 二级筛选（GPU优化版）
    print("\n开始GPU加速筛选...")
    for dc1, Nx, Ny, c, f in tqdm(level1_combos):
        # 计算二级变量范围
        dc2_min = 0.2 * 3 * dc1 / 8
        dc2_max = 1.6 * 3 * dc1 / 8
        lg_max = (Ny*c+1.4)/2
        dc2_vals = np.arange(max(dc2_min,1),  dc2_max + dc2_step / 2, dc2_step,dtype=np.float16)
        ht_vals = np.arange( max(3*dc2_min,1),  dc2_max + ht_step / 2, ht_step,dtype=np.float16)
        lg1_vals = np.arange(0.1, min(4,lg_max) , lg1_step,dtype=np.float16)

        # 转换为CuPy数组（确保不是元组）
        dc2_gpu = cp.asarray(dc2_vals,dtype=np.float16)
        ht_gpu = cp.asarray(ht_vals,dtype=np.float16)
        lg1_gpu = cp.asarray(lg1_vals,dtype=np.float16)

        # 生成网格（使用CuPy的meshgrid）
        dc2_mesh, ht_mesh, lg1_mesh = cp.meshgrid(
            dc2_gpu, ht_gpu, lg1_gpu,
            indexing='ij'
        )

        # 展平并计算i_max
        dc2_flat = dc2_mesh.ravel()
        ht_flat = ht_mesh.ravel()
        lg1_flat = lg1_mesh.ravel()
        i_max = (Bmax * lg1_flat*0.001) / (mu * Nx * Ny)

        # 生成所有可能的i值（GPU向量化）
        i_vals = cp.arange(1, 50.1, i_step, dtype=cp.float16)

        # 广播比较（完全GPU化）
        valid_i_mask = (i_vals[:, None] <= i_max) & (i_vals[:, None] >= 1)
        valid_positions = cp.any(valid_i_mask, axis=0)

        # 收集有效样本
        if cp.sum(valid_positions) > 0:
            # 获取有效参数
            valid_dc2 = dc2_flat[valid_positions]
            valid_ht = ht_flat[valid_positions]
            valid_lg1 = lg1_flat[valid_positions]

            # 获取对应的最小i值
            #valid_i = i_vals[valid_i_mask[:, valid_positions].argmax(axis=0)]
            # 获取对应的布尔掩码（只保留合法位置）
            valid_i_mask_selected = valid_i_mask[:, valid_positions]  # shape: [num_i, num_valid_combos]

            # 获取所有合法的 i 值，非法位置设置为 NaN（方便随机选择时忽略）
            i_vals_expanded = cp.broadcast_to(i_vals[:, None], valid_i_mask_selected.shape)
            i_vals_masked = cp.where(valid_i_mask_selected, i_vals_expanded, cp.nan)

            # 随机选择合法的 i 值
            rand_indices = cp.random.randint(0, i_vals_masked.shape[0], size=i_vals_masked.shape[1])
            chosen_i = i_vals_masked[rand_indices, cp.arange(i_vals_masked.shape[1])]

            # 如果随机到了 NaN（说明那个位置没有合法 i），就跳过这个样本
            valid_non_nan_mask = ~cp.isnan(chosen_i)

            # 过滤出有效的样本
            valid_dc2 = dc2_flat[valid_positions][valid_non_nan_mask]
            valid_ht = ht_flat[valid_positions][valid_non_nan_mask]
            valid_lg1 = lg1_flat[valid_positions][valid_non_nan_mask]
            valid_i = chosen_i[valid_non_nan_mask]


            # 转换为CPU数据（批量转换）
            samples = cp.stack([
                cp.full_like(valid_dc2, dc1),
                cp.full_like(valid_dc2, Nx),
                cp.full_like(valid_dc2, Ny),
                cp.full_like(valid_dc2, c),
                cp.full_like(valid_dc2, f),
                valid_dc2,
                valid_ht,
                valid_lg1,
                valid_i
            ], axis=1)

            valid_samples.extend(cp.asnumpy(samples).tolist())

    return valid_samples


# ------------------------ 主程序 ------------------------
if __name__ == "__main__":
    valid_samples = gpu_filter_combinations()
    print(f"\n最终有效组合数: {len(valid_samples)}")

    # 保存结果
    columns = ["dc1", "Nx", "Ny", "c", "f", "dc2", "ht", "lg1", "i"]
    df = pd.DataFrame(valid_samples, columns=columns)

    # 添加单位（单位拼接为字符串）
    df["dc1"] = df["dc1"].map(lambda x: f"{round(x, 1)}mm")
    df["Nx"] = df["Nx"].astype(int)  # 无单位
    df["Ny"] = df["Ny"].astype(int)  # 无单位
    df["c"] = df["c"].map(lambda x: f"{x}mm")
    df["f"] = df["f"].map(lambda x: f"{x}kHz")
    df["dc2"] = df["dc2"].map(lambda x: f"{round(x, 1)}mm")
    df["ht"] = df["ht"].map(lambda x: f"{round(x, 1)}mm")
    df["lg1"] = df["lg1"].map(lambda x: f"{round(x, 1)}mm")
    df["i"] = df["i"].map(lambda x: f"{round(x, 1)}A")
    # 2. 添加 mathindex（从原始 c 中解析）
    df["mathindex"] = df["c"].map(
        lambda
            x: f'copper_lizt_c_{int(float(x.replace("mm", ""))) if float(x.replace("mm", "")) % 1 == 0 else round(float(x.replace("mm", "")), 1)}'
    )

    for size in sample_sizes:
        if len(df) >= size:
            sample = df.sample(n=size, random_state=42).reset_index(drop=True)
            chunk_size = 500
            num_chunks = size // chunk_size

            for i in range(num_chunks):
                chunk = sample.iloc[i * chunk_size:(i + 1) * chunk_size]
                filename = os.path.join(output_dir, f"filtered_samples_{size}_{i + 1}.csv")
                chunk.to_csv(filename, index=False)
            print(f"已保存 {num_chunks} 个文件，每个包含 1000 个样本，总计 {size} 个样本")

            # ---------- 统计信息 ----------
            numeric_cols = ["dc1", "Nx", "Ny", "c", "f", "dc2", "ht", "lg1", "i"]
            # 去掉单位，统一转为 float32 再计算统计量
            stats_df = sample.copy()
            for col in numeric_cols:
                stats_df[col] = stats_df[col].astype(str).str.replace(r"[^\d\.]", "", regex=True).astype(np.float32)

            # 计算统计量
            summary = stats_df[numeric_cols].agg(["min", "max", "mean", "std"]).T
            summary = summary.round(4)
            summary.to_csv(os.path.join(output_dir, f"filtered_samples_{size}_stats.csv"))

            print(f"已保存统计信息文件：filtered_samples_{size}_stats.csv")

        else:
            print(f"警告: 只有 {len(df)} 个有效样本，无法生成 {size} 个样本的文件")

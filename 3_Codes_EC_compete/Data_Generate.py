# inputs.py
import numpy as np
import cupy as cp
import math
from tqdm import tqdm

# ------------------------ 物理常量 ------------------------
Bmax = 0.3  # T
mu = 4 * math.pi * 1e-7  # H/m

# ------------------------ 参数范围定义 ------------------------
dc1_range = np.arange(5, 30.1, 1)
#dc1_range =np.array([10, 9.6, 10.5, 11.4, 11.1, 12.5, 13.6,12.7,15.1,15.6,14.5,14.8,17.6,17.5,19,20.3])
Nx_range = np.arange(1, 10.1, 1)
Ny_range = np.arange(6, 6.1, 1)
c_range = np.arange(1,5.1, 0.5)
f_range = np.arange(200, 400, 300)
dc2_step = 0.2
ht_step = 1
lg1_step = 0.1
i_step = 5

def generate_all_valid_samples():
    valid_samples = []
    total_count = 0  # 添加计数器
    for dc1 in tqdm(dc1_range, desc="dc1"):
        for Nx in Nx_range:
            for Ny in Ny_range:
                for c in c_range:
                    if 1 < Nx * c < 20 and 0.1 < Ny * c < 20:
                    #if 5 < Nx * c < 9.5 and 0.1 < Ny * c < 20:
                        for f in f_range:
                            dc2_min = 0.2 * 3 * dc1 / 8
                            dc2_max = 1.6 * 3 * dc1 / 8
                            #dc2_vals = np.arange(max(dc2_min, 1), dc2_max + dc2_step / 2, dc2_step, dtype=np.float32)
                            dc2_vals = np.arange(np.pi * dc1 / 8, np.pi * dc1 / 8 + dc2_step / 2, dc2_step, dtype=np.float32)
                            #dc2_vals = np.array([3.3,4,4.1,4.6,4.8,5.1,5,5.5,5.8,6,5.6,6.9,6.1,7.2])
                            ht_vals = np.arange(1*np.pi * dc1 / 8,  np.pi * dc1 / 8+ht_step / 2 , ht_step, dtype=np.float32)
                            #ht_vals = np.arange(max(3 * dc2_min, 1), dc2_max + ht_step / 2, ht_step, dtype=np.float32)
                            lg1_vals = np.arange(0.1, 4, lg1_step)

                            #在GPU上生成三维网络
                            dc2_gpu = cp.asarray(dc2_vals)
                            ht_gpu = cp.asarray(ht_vals)
                            lg1_gpu = cp.asarray(lg1_vals)

                            #每个点就是(dc2,ht,lg1)的组合
                            dc2_mesh, ht_mesh, lg1_mesh = cp.meshgrid(dc2_gpu, ht_gpu, lg1_gpu, indexing='ij')
                            #.ravel转换为一维向量
                            dc2_flat = dc2_mesh.ravel()
                            ht_flat = ht_mesh.ravel()
                            lg1_flat = lg1_mesh.ravel()

                            #i_max = (Bmax * lg1_flat * 0.001) / (mu * Nx * Ny)
                            i_max = (Bmax * lg1_flat * 0.001) / (mu * Nx * Ny)
                            i_vals = cp.arange(1, 1.1, i_step, dtype=cp.float32)

                            valid_i_mask = (i_vals[:, None] <= i_max) & (i_vals[:, None] >= 1)
                            valid_positions = cp.any(valid_i_mask, axis=0)

                            if cp.sum(valid_positions) > 0:
                                valid_dc2 = dc2_flat[valid_positions]
                                valid_ht = ht_flat[valid_positions]
                                valid_lg1 = lg1_flat[valid_positions]

                                valid_i_mask_selected = valid_i_mask[:, valid_positions]
                                i_vals_expanded = cp.broadcast_to(i_vals[:, None], valid_i_mask_selected.shape)
                                i_vals_masked = cp.where(valid_i_mask_selected, i_vals_expanded, cp.nan)

                                rand_indices = cp.random.randint(0, i_vals_masked.shape[0], size=i_vals_masked.shape[1])
                                chosen_i = i_vals_masked[rand_indices, cp.arange(i_vals_masked.shape[1])]
                                valid_non_nan_mask = ~cp.isnan(chosen_i)

                                # 合法位置
                                valid_dc2 = valid_dc2[valid_non_nan_mask]
                                valid_ht = valid_ht[valid_non_nan_mask]
                                valid_lg1 = valid_lg1[valid_non_nan_mask]
                                valid_i = chosen_i[valid_non_nan_mask]

                                samples = cp.stack([
                                    cp.full_like(valid_dc2, dc1),       # dc1
                                    valid_dc2,                          # dc2
                                    valid_ht,                           # ht
                                    valid_lg1,                          # lg1
                                    cp.full_like(valid_dc2, Nx),        # Nx
                                    cp.full_like(valid_dc2, Ny),        # Ny
                                    cp.full_like(valid_dc2, c),         # c
                                    cp.full_like(valid_dc2, f),         # f
                                    valid_i                             # i
                                ], axis=1)

                                samples_list = cp.asnumpy(samples).tolist()
                                valid_samples.extend(cp.asnumpy(samples).tolist())
                                total_count += len(samples_list)
    print(f"\n📌 总组合数： {total_count}")  # 打印统计信息
    return valid_samples

# === 主函数：inputs() ===
def inputs(n=20, seed=42):
    """
    返回 n 个 (dc1, dc2, ht, lg1, Nx, Ny, c, f, i) 组合，满足筛选规则
    """
    all_valid = np.round(np.array(generate_all_valid_samples()), 2)  # ✅ 统一精度控制
    if n is None or n >= len(all_valid):
        return all_valid
    np.random.seed(seed)
    idx = np.random.choice(len(all_valid), n, replace=False)
    return [tuple(all_valid[i]) for i in idx]

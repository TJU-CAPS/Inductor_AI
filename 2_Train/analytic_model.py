# analytic_model.py
import numpy as np

def analytic_inductor_model(x_raw):
    """
    Analytic inductor model (vectorized), faithful to your original step-by-step formulas.
    Input:
        x_raw: numpy array, shape (N,9) or (9,)
               columns = [C(mm), dc1(mm), dc2(mm), f(kHz), ht(mm),
                          i(A), lg1(mm), Nx, Ny]
    Output:
        numpy array shape (N,3): [L(uH), Pw(mW), Pc(mW)]
    """

    x = np.asarray(x_raw, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # --- extract columns (units as provided) ---
    C_mm   = x[:, 0]    # mm
    dc1_mm = x[:, 1]    # mm
    dc2_mm = x[:, 2]    # mm
    f_khz  = x[:, 3]    # kHz
    ht_mm  = x[:, 4]    # mm
    i_amp  = x[:, 5]    # A (KEEP float!)
    lg1_mm = x[:, 6]    # mm
    Nx_arr = x[:, 7]    # turns (may be float in file)
    Ny_arr = x[:, 8]

    # --- unit conversion to SI ---
    c    = C_mm * 1e-3      # m
    dc1  = dc1_mm * 1e-3    # m
    dc2  = dc2_mm * 1e-3    # m
    f    = f_khz * 1e3      # Hz
    ht   = ht_mm * 1e-3     # m
    lg1  = lg1_mm * 1e-3    # m

    # treat Nx,Ny as integers for geometry sums (but keep arrays)
    Nx = np.round(Nx_arr).astype(int)
    Ny = np.round(Ny_arr).astype(int)

    # keep i as float (do NOT round to int)
    i = i_amp.astype(float)

    # constants and small epsilon
    mu = 4.0 * np.pi * 1e-7
    rho = 1.678e-8   # copper resistivity Ω·m
    eps = 1e-18

    # ---------------- 电感计算 ----------------
    # use lg = lg1/2 as original
    lg = lg1 / 2.0

    Ae1 = np.pi * (dc1 + 2.0*lg)**2 / 4.0
    Ae2 = (dc1 + 2.0*lg) * (dc2 + 2.0*lg)

    R1 = lg / (Ae1 * mu + eps)
    R2 = lg / (Ae2 * mu + eps)

    NxNy = (Nx_arr * Ny_arr).astype(float)
    L_H = (NxNy**2) / (R1 + R2 / 2.0 + eps)   # Henry
    L_uH = L_H * 1e6  # microhenry for output

    # print diagnostic (optional)
    # print("Debug L (uH):", L_uH)

    # ---------------- 绕组损耗 Pw ----------------
    # cross-sectional area of wire (m^2)
    S = np.pi * (c**2) / 4.0  # m^2

    # wire length l: vectorized closed form of the sum in original code
    # original: for i0 in range(Nx): l += 2*pi*(dc1/2 + 0.07e-3 + c/2 + i0*c)
    base_radius = dc1 / 2.0 + 0.07e-3 + c / 2.0  # m
    # sum_{i0=0}^{Nx-1} i0 = Nx*(Nx-1)/2
    l_wire = 2.0 * np.pi * ( Nx * base_radius + c * (Nx * (Nx - 1) / 2.0) )  # m

    Rdc = rho * l_wire / (S + eps)  # Ohm

    irms = i / np.sqrt(2.0)

    # skin depth df
    df = np.sqrt(2.0 * rho / (2.0 * np.pi * f * mu + eps))  # m

    # Rac correction F1 (you computed but didn't use F1 later; keep for completeness)
    Aw1 = Nx * Ny * c * c
    Aw2 = (Nx * c - 2.0 * df) * (Ny * c - 2.0 * df)
    # avoid division by zero
    F1 = Aw1 / (Aw2 + eps)

    # Rac correction F2 (original formula)
    arg = (Ny * c) / (df + eps)
    # clip arg to safe range to avoid hyperbolic overflow
    arg_clip = np.clip(arg, -200.0, 200.0)
    sinh_arg = np.sinh(arg_clip)
    sin_arg = np.sin(arg_clip)
    cosh_arg = np.cosh(arg_clip)
    cos_arg = np.cos(arg_clip)
    denom = (cosh_arg - cos_arg) + eps
    F2 = (Ny * c) / (df + eps) / 2.0 * ((sinh_arg + sin_arg) / denom)

    Pw_s = (irms**2) * Rdc * F2  # W

    # air-gap induced winding losses (follow original)
    core_area_mid = np.pi * (dc1**2) / 4.0
    core_area_edge = dc1 * dc2 * 2.0

    # Use L_H (H) in B computations, follow original formula
    # B1 = i*L/(Nx*Ny*np.pi*(dc1**2)/4)
    # B2 = i*L/(Nx*Ny*dc1*dc2*2)
    denom_mid = (Nx * Ny + eps) * (core_area_mid + eps)
    denom_edge = (Nx * Ny + eps) * (core_area_edge + eps)
    B1 = i * L_H / denom_mid
    B2 = i * L_H / denom_edge

    Pw_f1 = 0.7 * (1e-4) * lg * dc1 * np.pi * (B1**2) * (f**2)
    Pw_f2 = 0.7 * (1e-4) * lg * dc1 * 2.0 * (B2**2) * (f**2)

    Pw_W = Pw_s + Pw_f1 + Pw_f2


    # ---------------- 磁芯损耗 Pc ----------------
    # B3 = i*L/(Nx*Ny*dc1*ht*2)
    denom_B3 = (Nx * Ny + eps) * (dc1 * ht * 2.0 + eps)
    B3 = i * L_H / denom_B3

    # Pv (W/m^3)
    Pv1 = 0.5967 * (f**1.4942) * (B1**2.4413)
    Pv2 = 0.5967 * (f**1.4942) * (B2**2.4413)
    Pv3 = 0.5967 * (f**1.4942) * (B3**2.4413)

    # Volumes V1, V2, V3 (m^3)
    V1 = np.pi * (dc1**2) / 4.0 * (Ny * c - lg + 0.07e-3 * 2.0)
    V2 = (dc1 * dc2) * (Ny * c - lg + 0.07e-3 * 2.0) * 2.0
    V3 = (dc1 * ht) * ((Nx * c + 0.07e-3 * 2.0) * 2.0 + dc1) * 2.0

    # sanitize Pv arrays
    Pv1 = np.nan_to_num(Pv1, nan=0.0, posinf=0.0, neginf=0.0)
    Pv2 = np.nan_to_num(Pv2, nan=0.0, posinf=0.0, neginf=0.0)
    Pv3 = np.nan_to_num(Pv3, nan=0.0, posinf=0.0, neginf=0.0)

    Pc_W = Pv1 * V1 + Pv2 * V2 + Pv3 * V3

    # 确保每个输出都是二维 (N,1) 形式
    L_uH = np.atleast_2d(L_uH).reshape(-1, 1)
    Pw_W = np.atleast_2d(Pw_W).reshape(-1, 1)
    Pc_W = np.atleast_2d(Pc_W).reshape(-1, 1)

    # 横向拼接成 (N,3)
    out = np.hstack([L_uH, Pw_W, Pc_W])

    return out

"""
# quick test when run directly
if __name__ == "__main__":
    import numpy as np
    # test input: [C(mm), dc1(mm), dc2(mm), f(kHz), ht(mm), i(A), lg1(mm), Nx, Ny]
    x_test = np.array([
        [1.0, 15.1, 5.8, 200.0, 5.7, 3.0, 0.6, 7, 2],   # sample 1
        [1.0, 12.0, 6.0, 100.0, 6.0, 2.0, 0.5, 5, 3]    # sample 2
    ])
    y = analytic_inductor_model(x_test)
    print("L(uH), Pw(W), Pc(W):\n", y)
"""


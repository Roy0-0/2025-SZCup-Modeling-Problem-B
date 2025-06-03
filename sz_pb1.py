import numpy as np

def xy_to_xyz(x, y, Y=1.0):
    X = x / y * Y
    Z = (1 - x - y) / y * Y
    return np.array([X, Y, Z])

# ----------- Step 1: 输入三基色与白点的色度坐标 -----------

# 显示设备（如BT.709）的三原色坐标（根据图中红色三角形）
xr, yr = 0.64, 0.33
xg, yg = 0.30, 0.60
xb, yb = 0.15, 0.06

# 白点（一般采用D65）
xw, yw = 0.3127, 0.3290

# ----------- Step 2: 将 xy 转换为 XYZ -----------

Xr = xy_to_xyz(xr, yr)
Xg = xy_to_xyz(xg, yg)
Xb = xy_to_xyz(xb, yb)
Xw = xy_to_xyz(xw, yw)

# ----------- Step 3: 构造 RGB → XYZ 的基础矩阵 -----------

# 构建矩阵 [Xr Xg Xb]（每列是一个三刺激值）
RGB_XYZ = np.stack([Xr, Xg, Xb], axis=1)

# 求解缩放因子，使得 RGB = [1, 1, 1] 时 → 白点 XYZ
S = np.linalg.inv(RGB_XYZ) @ Xw  # S 为缩放因子（对 R/G/B 通道）

# 最终的 RGB → XYZ 转换矩阵
M_RGB2XYZ = RGB_XYZ @ np.diag(S)

# ----------- Step 4: 得到 XYZ → RGB 的反变换矩阵 -----------

M_XYZ2RGB = np.linalg.inv(M_RGB2XYZ)

# ----------- Step 5: 输出 -----------

print("RGB → XYZ 转换矩阵:")
print(M_RGB2XYZ)

print("\nXYZ → RGB 反变换矩阵:")
print(M_XYZ2RGB)
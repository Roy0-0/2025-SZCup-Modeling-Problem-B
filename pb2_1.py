import numpy as np

def xyY_to_XYZ(x, y, Y=1.0):
    X = (x / y) * Y
    Z = ((1 - x - y) / y) * Y
    return np.array([X, Y, Z])

# 示例输入：RGBV（输入）和RGBCX（输出）各通道的 xyY

rgbv_xy = {
    'R': (0.708, 0.292),
    'G': (0.170, 0.797),
    'B': (0.131, 0.046),
    'V': (0.20, 0.10)
}
rgbcx_xy = {
    'R': (0.64, 0.33),
    'G': (0.30, 0.60),
    'B': (0.15, 0.06),
    'C': (0.22, 0.33),
    'X': (0.42, 0.51)
}

# 构建基色矩阵
M_in = np.column_stack([xyY_to_XYZ(x, y) for x, y in rgbv_xy.values()])    # 3x4
M_out = np.column_stack([xyY_to_XYZ(x, y) for x, y in rgbcx_xy.values()])  # 3x5

# 求解 T: 5x4 映射矩阵
# 解 M_out @ T = M_in → T = (M_out^T M_out)^-1 M_out^T M_in
T = np.linalg.lstsq(M_out, M_in, rcond=None)[0] # T: 5x4

print("线性映射矩阵 T (5x4):")
print(T)

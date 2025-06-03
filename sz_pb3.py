import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def rgb2xyz(r, g, b):
    r /= 255.0
    g /= 255.0
    b /= 255.0

    # gamma校正
    r = pow(r, 2.2) if r > 0.04045 else r / 12.92
    g = pow(g, 2.2) if g > 0.04045 else g / 12.92
    b = pow(b, 2.2) if b > 0.04045 else b / 12.92

    # sRGB->XYZ 基于问题一
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    return x, y, z

def xyz2rgb(x, y, z):
    # XYZ->sRGB 基于问题一
    r = x * 3.2410 + y * -1.5374 + z * -0.4986
    g = x * -0.9692 + y * 1.8760 + z * 0.0415
    b = x * 0.0556 + y * -0.2040 + z * 1.0570

    # 逆gamma校正
    r = 1.055 * pow(r, 1 / 2.4) - 0.055 if r > 0.0031308 else r * 12.92
    g = 1.055 * pow(g, 1 / 2.4) - 0.055 if g > 0.0031308 else g * 12.92
    b = 1.055 * pow(b, 1 / 2.4) - 0.055 if b > 0.0031308 else b * 12.92

    r = max(0, min(1, r))
    g = max(0, min(1, g))
    b = max(0, min(1, b))

    return int(r * 255), int(g * 255), int(b * 255)

def color_distance(rgb1, rgb2):
    return np.sqrt(np.sum((np.array(rgb1) - np.array(rgb2)) ** 2))

def load_data(filepath):
    try:
        target_data = pd.read_excel(filepath, sheet_name='RGB目标值')
        r_r = pd.read_excel(filepath, sheet_name='R_R')
        r_g = pd.read_excel(filepath, sheet_name='R_G')
        r_b = pd.read_excel(filepath, sheet_name='R_B')
        g_r = pd.read_excel(filepath, sheet_name='G_R')
        g_g = pd.read_excel(filepath, sheet_name='G_G')
        g_b = pd.read_excel(filepath, sheet_name='G_B')
        b_r = pd.read_excel(filepath, sheet_name='B_R')
        b_g = pd.read_excel(filepath, sheet_name='B_G')
        b_b = pd.read_excel(filepath, sheet_name='B_B')

        return {
            'target': target_data,
            'red': (r_r, r_g, r_b),
            'green': (g_r, g_g, g_b),
            'blue': (b_r, b_g, b_b)
        }
    except Exception as e:
        print(f"加载出错：{e}")
        return None

def color_correction(data):
    target_data = data['target']
    red_data = data['red']
    green_data = data['green']
    blue_data = data['blue']

    rows, cols = target_data.shape
    corrected_red = np.zeros((rows, cols, 3), dtype=np.uint8)
    corrected_green = np.zeros((rows, cols, 3), dtype=np.uint8)
    corrected_blue = np.zeros((rows, cols, 3), dtype=np.uint8)

    target_red = (220, 0, 0)
    target_green = (0, 220, 0)
    target_blue = (0, 0, 220)

    target_red_xyz = rgb2xyz(*target_red)
    target_green_xyz = rgb2xyz(*target_green)
    target_blue_xyz = rgb2xyz(*target_blue)

    for i in range(rows):
        for j in range(cols):
            current_red = (
                red_data[0].iloc[i, j],
                red_data[1].iloc[i, j],
                red_data[2].iloc[i, j]
            )
            current_green = (
                green_data[0].iloc[i, j],
                green_data[1].iloc[i, j],
                green_data[2].iloc[i, j]
            )
            current_blue = (
                blue_data[0].iloc[i, j],
                blue_data[1].iloc[i, j],
                blue_data[2].iloc[i, j]
            )
            current_red_xyz = rgb2xyz(*current_red)
            current_green_xyz = rgb2xyz(*current_green)
            current_blue_xyz = rgb2xyz(*current_blue)

            # 比例系数
            red_correction = [t / c if c > 0 else 1 for t, c in zip(target_red_xyz, current_red_xyz)]
            green_correction = [t / c if c > 0 else 1 for t, c in zip(target_green_xyz, current_green_xyz)]
            blue_correction = [t / c if c > 0 else 1 for t, c in zip(target_blue_xyz, current_blue_xyz)]

            corrected_red_xyz = [c * f for c, f in zip(current_red_xyz, red_correction)]
            corrected_green_xyz = [c * f for c, f in zip(current_green_xyz, green_correction)]
            corrected_blue_xyz = [c * f for c, f in zip(current_blue_xyz, blue_correction)]

            corrected_red[i, j] = xyz2rgb(*corrected_red_xyz)
            corrected_green[i, j] = xyz2rgb(*corrected_green_xyz)
            corrected_blue[i, j] = xyz2rgb(*corrected_blue_xyz)

    return corrected_red, corrected_green, corrected_blue

def visualizing(original_red, original_green, original_blue, corrected_red, corrected_green, corrected_blue):
    """可视化校正前后的图像对比"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('颜色校正结果对比', fontsize=16, fontweight='bold')

    axes[0, 0].imshow(original_red)
    axes[0, 0].set_title('校正前红色通道')
    axes[0, 1].imshow(original_green)
    axes[0, 1].set_title('校正前绿色通道')
    axes[0, 2].imshow(original_blue)
    axes[0, 2].set_title('校正前蓝色通道')
    axes[1, 0].imshow(corrected_red)
    axes[1, 0].set_title('校正后红色通道')
    axes[1, 1].imshow(corrected_green)
    axes[1, 1].set_title('校正后绿色通道')
    axes[1, 2].imshow(corrected_blue)
    axes[1, 2].set_title('校正后蓝色通道')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def analyze_color_difference(original_red, original_green, original_blue,
                           corrected_red, corrected_green, corrected_blue):
    """
    分析并可视化色差改善情况
    """
    # 目标颜色
    target_red = (220, 0, 0)
    target_green = (0, 220, 0)
    target_blue = (0, 0, 220)

    # 计算色差改善的函数
    def calculate_improvement(original, corrected, target):
        original_dist = color_distance(original, target)
        corrected_dist = color_distance(corrected, target)
        improvement = (original_dist - corrected_dist) / original_dist * 100 if original_dist > 0 else 0
        return original_dist, corrected_dist, improvement

    # 收集色差数据
    results = []
    rows, cols, _ = original_red.shape

    for i in range(rows):
        for j in range(cols):
            # 红色通道
            orig_r = original_red[i, j]
            corr_r = corrected_red[i, j]
            orig_dist_r, corr_dist_r, imp_r = calculate_improvement(orig_r, corr_r, target_red)

            # 绿色通道
            orig_g = original_green[i, j]
            corr_g = corrected_green[i, j]
            orig_dist_g, corr_dist_g, imp_g = calculate_improvement(orig_g, corr_g, target_green)

            # 蓝色通道
            orig_b = original_blue[i, j]
            corr_b = corrected_blue[i, j]
            orig_dist_b, corr_dist_b, imp_b = calculate_improvement(orig_b, corr_b, target_blue)

            results.append({
                'row': i,
                'col': j,
                'red_original_dist': orig_dist_r,
                'red_corrected_dist': corr_dist_r,
                'red_improvement': imp_r,
                'green_original_dist': orig_dist_g,
                'green_corrected_dist': corr_dist_g,
                'green_improvement': imp_g,
                'blue_original_dist': orig_dist_b,
                'blue_corrected_dist': corr_dist_b,
                'blue_improvement': imp_b
            })

    # 创建DataFrame
    df = pd.DataFrame(results)

    # 创建报告目录
    report_dir = "color_correction_report"
    os.makedirs(report_dir, exist_ok=True)

    # 保存原始数据到CSV
    csv_path = os.path.join(report_dir, "color_difference_data.csv")
    df.to_csv(csv_path, index=False)

    # 创建色差改善可视化图表
    plt.figure(figsize=(15, 10))

    # 解决零方差问题的绘图函数
    def safe_kdeplot(data, label, color, ax):
        if data.nunique() > 1:  # 只有数据有变化时才绘制KDE
            sns.kdeplot(data, label=label, color=color, fill=True, ax=ax)
        else:
            # 添加一条垂直线表示常数值
            ax.axvline(x=data.iloc[0], color=color, linestyle='--', label=f'{label} (常数值)')

    # 色差分布对比
    ax1 = plt.subplot(2, 2, 1)
    safe_kdeplot(df['red_original_dist'], '校正前红色', 'red', ax1)
    safe_kdeplot(df['red_corrected_dist'], '校正后红色', 'darkred', ax1)
    safe_kdeplot(df['green_original_dist'], '校正前绿色', 'green', ax1)
    safe_kdeplot(df['green_corrected_dist'], '校正后绿色', 'darkgreen', ax1)
    safe_kdeplot(df['blue_original_dist'], '校正前蓝色', 'blue', ax1)
    safe_kdeplot(df['blue_corrected_dist'], '校正后蓝色', 'darkblue', ax1)
    plt.title('色差分布对比')
    plt.xlabel('色差值')
    plt.ylabel('密度')
    plt.legend()

    # 改善率分布
    plt.subplot(2, 2, 2)
    sns.histplot(df[['red_improvement', 'green_improvement', 'blue_improvement']],
                kde=True, bins=20, alpha=0.5)
    plt.title('各通道色差改善率分布')
    plt.xlabel('改善率(%)')
    plt.ylabel('像素数量')

    # 改善率箱线图
    plt.subplot(2, 2, 3)
    improvement_df = df[['red_improvement', 'green_improvement', 'blue_improvement']].melt(
        var_name='通道', value_name='改善率')
    sns.boxplot(x='通道', y='改善率', data=improvement_df)
    plt.title('各通道色差改善率箱线图')

    # 平均色差对比
    plt.subplot(2, 2, 4)
    avg_data = pd.DataFrame({
        '通道': ['红色', '绿色', '蓝色'],
        '校正前': [df['red_original_dist'].mean(), df['green_original_dist'].mean(), df['blue_original_dist'].mean()],
        '校正后': [df['red_corrected_dist'].mean(), df['green_corrected_dist'].mean(), df['blue_corrected_dist'].mean()]
    })
    avg_data = avg_data.melt(id_vars='通道', var_name='状态', value_name='平均色差')
    sns.barplot(x='通道', y='平均色差', hue='状态', data=avg_data)
    plt.title('平均色差对比')

    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'color_difference_analysis.png'), dpi=300)
    plt.close()

    return df

def main():
    # 1. 加载数据
    file_path = "pb_3.xlsx"
    data = load_data(file_path)

    if data is None:
        print("无法加载数据，请检查文件路径和格式")
        return

    # 2. 准备原始图像数据
    target_data = data['target']
    red_data = data['red']
    green_data = data['green']
    blue_data = data['blue']

    rows, cols = target_data.shape

    original_red = np.zeros((rows, cols, 3), dtype=np.uint8)
    original_green = np.zeros((rows, cols, 3), dtype=np.uint8)
    original_blue = np.zeros((rows, cols, 3), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            original_red[i, j] = [
                red_data[0].iloc[i, j],
                red_data[1].iloc[i, j],
                red_data[2].iloc[i, j]
            ]
            original_green[i, j] = [
                green_data[0].iloc[i, j],
                green_data[1].iloc[i, j],
                green_data[2].iloc[i, j]
            ]
            original_blue[i, j] = [
                blue_data[0].iloc[i, j],
                blue_data[1].iloc[i, j],
                blue_data[2].iloc[i, j]
            ]

    # 3. 执行颜色校正
    print("正在进行颜色校正...")
    corrected_red, corrected_green, corrected_blue = color_correction(data)
    print("颜色校正完成!")

    # 4. 可视化校正前后对比
    print("生成校正结果对比图...")
    fig = visualizing(original_red, original_green, original_blue,
                     corrected_red, corrected_green, corrected_blue)
    fig.savefig('color_correction_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 5. 执行色差分析
    print("执行色差分析...")
    color_diff_df = analyze_color_difference(
        original_red, original_green, original_blue,
        corrected_red, corrected_green, corrected_blue
    )

    print("处理完成！")

if __name__ == "__main__":
    main()
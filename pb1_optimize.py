import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from time import time
from torch.amp import autocast, GradScaler  # 导入正确的混合精度模块


# -------------------- 1. 数据生成与增强 --------------------
def generate_xyz_samples(n=50000, augment=True):
    """生成和增强XYZ色域样本，并计算对应的RGB值"""
    # 定义RGB到XYZ的转换矩阵（用于生成数据）
    M_RGB2XYZ = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])

    # 生成随机RGB值作为基础样本
    rgb_samples = np.random.rand(n, 3)

    # 转换为XYZ空间
    xyz_samples = np.dot(rgb_samples, M_RGB2XYZ.T)

    # 标准化到[0,1]范围
    xyz_min = xyz_samples.min(axis=0)
    xyz_max = xyz_samples.max(axis=0)
    xyz_normalized = (xyz_samples - xyz_min) / (xyz_max - xyz_min)

    # 数据增强（添加随机噪声和扰动）
    if augment:
        noise = np.random.normal(0, 0.005, xyz_normalized.shape)
        xyz_normalized = np.clip(xyz_normalized + noise, 0, 1)

    # 计算对应的RGB值（用于训练目标）
    M_XYZ2RGB = np.linalg.inv(M_RGB2XYZ)
    rgb_targets = np.dot(xyz_normalized, M_XYZ2RGB.T)
    rgb_targets = np.clip(rgb_targets, 0, 1)  # 确保RGB值在[0,1]范围内

    return (torch.tensor(xyz_normalized, dtype=torch.float32),
            torch.tensor(rgb_targets, dtype=torch.float32))


# -------------------- 2. 颜色转换矩阵 --------------------
M_XYZ2RGB = torch.tensor([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252]
], dtype=torch.float32)


def xyz_to_rgb(xyz):
    """XYZ到RGB的转换，支持批量处理"""
    device = xyz.device
    if not hasattr(xyz_to_rgb, 'M_device') or xyz_to_rgb.M_device.device != device:
        xyz_to_rgb.M_device = M_XYZ2RGB.to(device)
    return torch.mm(xyz, xyz_to_rgb.M_device.T)


# -------------------- 3. 改进的神经网络模型 --------------------
class ColorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 3),
            nn.Sigmoid()  # 确保输出在[0,1]范围内
        )
        # 权重初始化
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


# -------------------- 4. 优化的训练流程 --------------------
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据准备 - 生成XYZ和对应的RGB数据
    XYZ_data, RGB_targets = generate_xyz_samples(50000)
    train_loader = DataLoader(TensorDataset(XYZ_data, RGB_targets),
                              batch_size=512,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=2)

    model = ColorNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-6)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=1000,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )

    scaler = GradScaler(enabled=device.type == 'cuda')
    losses = []
    start_time = time()
    best_loss = float('inf')

    for epoch in range(1000):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            XYZ, RGB_target = batch
            XYZ = XYZ.to(device, non_blocking=True)
            RGB_target = RGB_target.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(device_type='cuda', enabled=device.type == 'cuda'):
                RGB_pred = model(XYZ)
                loss = nn.functional.mse_loss(RGB_pred, RGB_target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        scheduler.step()

        if epoch % 50 == 0 or epoch == 999:
            lr = optimizer.param_groups[0]['lr']
            elapsed = (time() - start_time) / 60
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.6f} | LR: {lr:.2e} | Time: {elapsed:.1f}min")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), "best_model.pth")
                print(f"Saved best model at epoch {epoch} with loss: {avg_loss:.6f}")

    return model, losses


# -------------------- 5. 可视化 --------------------
def visualize_results(model, losses):
    plt.figure(figsize=(15, 5))

    # 损失曲线
    plt.subplot(131)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # 色度图
    model.eval()
    with torch.no_grad():
        XYZ_test, RGB_test = generate_xyz_samples(1000, augment=False)
        XYZ_test = XYZ_test.to(next(model.parameters()).device)
        RGB_test = RGB_test.cpu()

        RGB_pred = model(XYZ_test).cpu()

        # 计算预测的XYZ值（用于比较）
        XYZ_pred = xyz_to_rgb(RGB_pred).cpu()

        def to_xy(XYZ):
            sum_XYZ = XYZ.sum(dim=1, keepdim=True) + 1e-6
            return XYZ[:, :2] / sum_XYZ

        xy_input = to_xy(XYZ_test.cpu())
        xy_pred = to_xy(XYZ_pred)

        plt.subplot(132)
        plt.scatter(xy_input[:, 0], xy_input[:, 1], s=2, c=RGB_test, alpha=0.3)
        plt.title("Input Chromaticity (XYZ)")

        plt.subplot(133)
        plt.scatter(xy_pred[:, 0], xy_pred[:, 1], s=2, c=RGB_pred, alpha=0.3)
        plt.title("Predicted Chromaticity (XYZ)")

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150)
    plt.show()


# -------------------- 主程序 --------------------
if __name__ == "__main__":
    model, losses = train_model()
    torch.save(model.state_dict(), "final_model.pth")
    visualize_results(model, losses)
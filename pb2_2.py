import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset


class EnhancedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 5),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def create_data_loaders(X_train, Y_train, X_val, Y_val, batch_size=32):
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2)
    return train_loader, val_loader


def train_and_validate(model, train_loader, val_loader, optimizer, scheduler, loss_fn, M_out_tensor, epochs=500):
    best_loss = float('inf')
    patience = 30
    counter = 0
    history = {'train': [], 'val': []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            pred_xyz = torch.matmul(outputs, M_out_tensor)
            loss = loss_fn(pred_xyz, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                outputs = model(X_batch)
                pred_xyz = torch.matmul(outputs, M_out_tensor)
                val_loss += loss_fn(pred_xyz, Y_batch).item()

        # Record metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Progress monitoring
        if epoch % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {lr:.2e}")

    return history


def main():
    # 初始化设置
    if not os.path.exists('images'):
        os.makedirs('images')

    # 数据准备 (示例数据，实际使用时替换为真实数据)
    np.random.seed(42)
    X = np.random.rand(1000, 4)  # 增加数据量
    Y = np.column_stack([
        X[:, 0] * 0.3 + X[:, 1] * 0.6 + np.random.normal(0, 0.02, 1000),
        X[:, 1] * 0.7 + X[:, 2] * 0.2 + np.random.normal(0, 0.02, 1000),
        X[:, 2] * 0.4 + X[:, 3] * 0.5 + np.random.normal(0, 0.02, 1000)
    ])

    # 数据划分
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]

    # 转换为张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)

    # 转换矩阵 (示例值，实际使用时替换为真实矩阵)
    M_out = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
        [0.3000, 0.4000, 0.1000],
        [0.2000, 0.1000, 0.5000]
    ])
    M_out_tensor = torch.tensor(M_out, dtype=torch.float32)

    # 数据加载器
    train_loader, val_loader = create_data_loaders(X_train, Y_train, X_val, Y_val)

    # 模型初始化
    model = EnhancedModel()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5, verbose=True)
    loss_fn = nn.MSELoss()

    # 训练与验证
    history = train_and_validate(model, train_loader, val_loader, optimizer, scheduler, loss_fn, M_out_tensor)

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))

    # 可视化结果 (保持与之前相同的可视化)
    plt.figure(figsize=(15, 5))

    # 1. 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)

    # 2. 预测对比
    with torch.no_grad():
        pred = model(X_train)
        xyz_pred = torch.matmul(pred, M_out_tensor).numpy()

        plt.subplot(1, 3, 2)
        for i, color in enumerate(['r', 'g', 'b']):
            plt.scatter(Y_train[:, i], xyz_pred[:, i], c=color, alpha=0.6,
                        label=f'{"XYZ"[i]} channel')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect prediction')
        plt.title('Predicted vs True Values')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True)

    # 3. 误差分布
    errors = xyz_pred - Y_train.numpy()
    plt.subplot(1, 3, 3)
    plt.boxplot(errors, labels=['X', 'Y', 'Z'])
    plt.title('Prediction Error Distribution')
    plt.ylabel('Error')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('images/optimized_results.png')
    plt.close()

    print("训练完成，可视化结果已保存到images/optimized_results.png")


if __name__ == "__main__":
    main()
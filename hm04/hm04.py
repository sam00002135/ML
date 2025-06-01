import torch
import torch.nn as nn
import torch.nn.functional as F

seven_segment_outputs = torch.tensor([
    [1, 1, 1, 1, 1, 1, 0],  # 0
    [0, 1, 1, 0, 0, 0, 0],  # 1
    [1, 1, 0, 1, 1, 0, 1],  # 2
    [1, 1, 1, 1, 0, 0, 1],  # 3
    [0, 1, 1, 0, 0, 1, 1],  # 4
    [1, 0, 1, 1, 0, 1, 1],  # 5
    [1, 0, 1, 1, 1, 1, 1],  # 6
    [1, 1, 1, 0, 0, 0, 0],  # 7
    [1, 1, 1, 1, 1, 1, 1],  # 8
    [1, 1, 1, 1, 0, 1, 1],  # 9
], dtype=torch.float32)

# one-hot 輸入資料
X = torch.eye(10)
Y = seven_segment_outputs

# 簡單 MLP 模型


class SevenSegmentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# 初始化模型
model = SevenSegmentModel()

# 損失函數
criterion = nn.BCELoss()

# 學習率
lr = 0.1

# 訓練迴圈
for epoch in range(1000):
    # 前向傳播
    output = model(X)
    loss = criterion(output, Y)

    # 清空舊梯度
    model.zero_grad()

    # 反向傳播（計算梯度）
    loss.backward()

    # 手動更新每一個參數 (梯度下降法)
    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad

    # 每 100 回合顯示一次損失
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

# 測試模型
print("\nPredictions:")
with torch.no_grad():
    preds = model(X).round()
    for i in range(10):
        print(f"Input: {i} -> Predicted Segments: {preds[i].tolist()}")

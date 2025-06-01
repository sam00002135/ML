import torch
import matplotlib.pyplot as plt

# 產生測試資料 (y = 3x + 2 + noise)
torch.manual_seed(0)
x = torch.linspace(0, 10, 100).unsqueeze(1)  # shape: (100, 1)
true_w = 3.0
true_b = 2.0
noise = torch.randn_like(x) * 0.5
y = true_w * x + true_b + noise  # shape: (100, 1)

# 定義線性模型
model = torch.nn.Linear(1, 1)  # 一個輸入，一個輸出

# 定義損失函數 (均方誤差)
loss_fn = torch.nn.MSELoss()

# 使用 SGD 優化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 訓練模型
for epoch in range(300):
    # 預測
    y_pred = model(x)

    # 計算損失
    loss = loss_fn(y_pred, y)

    # 清空梯度
    optimizer.zero_grad()

    # 反向傳播
    loss.backward()

    # 更新參數
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# 顯示結果
w, b = model.parameters()
print(f"\nLearned weight: {w.item():.4f}, bias: {b.item():.4f}")

# 畫圖觀察結果
plt.scatter(x.numpy(), y.numpy(), label='Data')
plt.plot(x.numpy(), y_pred.detach().numpy(), color='red', label='Fitted line')
plt.legend()
plt.show()

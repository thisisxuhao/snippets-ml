import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义一些数据用于进行线性回归
x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y_train = np.dot(x_train, np.array([1, 2])) + 3

# 转换为Tensor
x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

# 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression(2, 1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每100次迭代输出一次结果
    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# 预测新数据
x_new_tensor = torch.Tensor([[3, 5]])
y_new_tensor = model(x_new_tensor)
y_new = y_new_tensor.item()
print('Predicted value of y for x = [3, 5]:', y_new)

# 创建一个三维图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制数据点
ax.scatter(x_train[:,0], x_train[:,1], y_train)

# 绘制线性回归平面
xx, yy = np.meshgrid(range(3), range(6))
print(xx.shape)
print(yy.shape)
zz = np.array([[model(torch.Tensor([a, b])).item() for a, b in zip(i, j)] for i, j in zip(xx, yy)])
print(zz)
# zz = np.array([model(torch.Tensor([[i, j]])).item() for i, j in zip(xx, yy)])
print(zz.shape)
ax.plot_surface(xx, yy, zz, alpha=0.5)

# 显示图像
plt.show()

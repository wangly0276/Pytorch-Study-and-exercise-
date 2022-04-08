import torch
import matplotlib.pyplot as plt  # 图形化
from torch import nn,optim
import numpy as np
learning_data = 0.01  # 学习率
# 1,准备数据
# y = 3x+0.8
x = torch.rand([500,1])
y = x * 3 + 0.8

# 2，定义模型
'''w = torch.rand([1,1],requires_grad=True)
b = torch.tensor(0,requires_grad=True,dtype=torch.float32)
'''
class Lr (nn.Module):
    def __init__(self):
        super(Lr, self).__init__()  # 继承父类init的参数
        self.linear = nn.Linear(1, 1)  # 1层
        self.linear = self.linear  # 2层
        # 可增加多层

    def forward(self, x):
        out = self.linear(x)
        return out

# 3,实例化模型,loss,优化器
model = Lr()
criterion = nn.MSELoss()
potimizer = optim.SGD(model.parameters(),lr=learning_data)  # 实例化
# 4，通过循环，反向传播，更新参数
# 训练模型
for i in range(3000):
    out = model(x)  # 获取预测值
    loss = criterion(y, out)  # 计算损失criterion
    potimizer.zero_grad()  # 梯度归0
    loss.backward()  # 计算梯度
    potimizer.step()  # 更新梯度
    if (i + 1) % 20 == 0 :
        print('Epoch[{}/{}],loss:{:6f}',format(loss.data,str(i)))

# 5,模型评估
model.eval()
predict = model(x)
predict = predict.data.numpy()
plt.scatter(x.data.numpy(), y.data.numpy(),c='r')
plt.plot(x.data.numpy(),predict)
plt.show()

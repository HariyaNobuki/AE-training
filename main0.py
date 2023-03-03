import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

# これはAEではないという認識でOK?

# make random 
torch.manual_seed(0)

x = torch.tensor([[1., 2., 3.]]) #入力値
label = torch.tensor([[1.]]) #正解ラベル

model = nn.Sequential(
    nn.Linear(3, 2), #入力3次元、出力2次元の全結合層
    nn.ReLU(inplace=True), #活性化関数
    nn.Linear(2, 1), #入力2次元、出力1次元の全結合層
)

print(f'x:{x}, label:{label}, pred:{model(x)}')
print(f'x.shape:{x.shape}, label.shape:{label.shape}, pred.shape:{model(x).shape}')
# visualization
print(summary(model, input_size=(1, 3)))

lr = 0.01
criterion = nn.MSELoss() #損失関数:平均二乗誤差
optimizer = optim.Adam(model.parameters(), lr=lr)
model, x = model.to('cpu'), x.to('cpu') #モデルと入力をCPUに移動

for epoch in range(200):
    optimizer.zero_grad() #勾配の初期化
    y = model(x) #順伝播
    loss = criterion(y, label) #損失の計算
    loss.backward() #誤差逆伝播
    optimizer.step() #パラメータの更新
    if epoch % 20 == 0:
        print(f'epoch: {epoch}, loss: {loss:.3f}')
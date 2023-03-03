import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

torch.manual_seed(0) #乱数の固定

x = torch.tensor([[1., 2., 3.]]) #入力値　※labelデータは無し(xがlabelになる)
label = x.detach()  # 独立した別のテンソルを作成

AutoE = nn.Sequential(
    nn.Linear(3, 2), #入力3次元、出力2次元の全結合層
    nn.ReLU(inplace=True), #活性化関数
    nn.Linear(2, 3), #入力2次元、出力1次元の全結合層
)

print(f'初期値| x:{x}, label:{label}, pred:{AutoE(x)}')
print(f'x.shape:{x.shape}, label.shape:{label.shape}, pred.shape:{AutoE(x).shape}')
print(summary(AutoE, input_size=(1, 3)))

lr = 0.01
criterion = nn.MSELoss() #損失関数:平均二乗誤差
optimizer = optim.Adam(AutoE.parameters(), lr=lr) #最適化手法:Adam
AutoE, x = AutoE.to('cpu'), x.to('cpu') #モデルと入力をCPUに移動

for epoch in range(1000):
    optimizer.zero_grad() #勾配の初期化
    y = AutoE(x) #順伝播
    loss = criterion(y, x) #損失の計算　※label=x
    loss.backward() #誤差逆伝播
    optimizer.step() #パラメータの更新
    if epoch % 100 == 0:
        print(f'epoch: {epoch}, loss: {loss:.3f}')
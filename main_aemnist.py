import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# トランスフォームオブジェクトを生成
# torchしていて不可解なやつ
transform = transforms.Compose(
    [transforms.ToTensor(), # Tensorオブジェクトに変換
     nn.Flatten()]) # データの形状を(28, 28)から784,)に変換

# MNISTの訓練用データ
mnist_train = torchvision.datasets.MNIST(
    root='MNIST',
    download=True, # ダウンロードを許可
    train=True,    # 訓練データを指定
    transform=transform) # トランスフォームオブジェクト

# MNISTのテスト用データ
mnist_test = torchvision.datasets.MNIST(
    root='MNIST',
    download=True, # ダウンロードを許可
    train=False,   # テストデータを指定
    transform=transform) # トランスフォームオブジェクト

# データローダーを生成
train_dataloader = DataLoader(mnist_train,    # 訓練データ
                              batch_size=124, # ミニバッチのサイズ
                              shuffle=True)   # 抽出時にシャッフル
test_dataloader = DataLoader(mnist_test,     # テストデータ
                              batch_size=1,  # テストなので1
                              shuffle=False) # 抽出時にシャッフルしない

import matplotlib.pyplot as plt

# type of tuple
_ = mnist_train[0]
# flatten size
print(f'type:{type(_)}, data:{_[0].shape}, label:{_[1]}')
print(f'最大値:{_[0].max()}, 最小値:{_[0].min()}')
plt.imshow(_[0].reshape(28, 28), cmap='gray')

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # torchはネットワークの話を記述する
        self.l1 = nn.Linear(784, 200) # エンコーダー(200ユニット)
        self.l2 = nn.Linear(200, 784) # デコーダー(784ユニット)

    def forward(self, x):
        h = self.l1(x)       # エンコーダーに入力
        h = torch.relu(h)    # ReLU関数を適用

        h = self.l2(h)       # デコーダーに入力
        y = torch.sigmoid(h) # シグモイド関数を適用
        return y

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

import torch.optim as optimizers

model = Autoencoder().to(device) # オートエンコーダーを生成
criterion = nn.BCELoss() # 損失関数はバイナリクロスエントロピー誤差
optimizer = optimizers.Adam(model.parameters()) # オプティマイザー（最適化関数）をAdamに設定

epochs = 10 # エポック数

# 学習の実行
for epoch in range(epochs):
    train_loss = 0.
    # ミニバッチのループ(ステップ)
    for (x, _) in train_dataloader:
        # 毎回書き直すのは理解できないがおそらく仕様
        x = x.to(device) # デバイスの割り当て
        model.train()    # 訓練モードにする
        preds = model(x) # モデルの出力を取得
        loss = criterion(preds, x) # 入力xと復元predsの誤差を取得
        optimizer.zero_grad()      # 勾配を0で初期化
        loss.backward()  # 誤差の勾配を計算
        optimizer.step() # パラメーターの更新
        train_loss += loss.item() # 誤差(損失)の更新
    # 1エポックあたりの損失を求める
    train_loss /= len(train_dataloader)
    # 1エポックごとに損失を出力
    print('Epoch({}) -- Loss: {:.3f}'.format(
        epoch+1,
        train_loss
    ))

import matplotlib.pyplot as plt

# テストデータを1個取り出す
_x, _ = next(iter(test_dataloader))
_x = _x.to(device)

model.eval() # ネットワークを評価モードにする
x_rec = model(_x) # テストデータを入力して結果を取得

# 入力画像、復元画像を表示
titles = {0: 'Original', 1: 'Autoencoder:Epoch=10'}
for i, image in enumerate([_x, x_rec]):
    image = image.view(28, 28).detach().cpu().numpy()
    plt.subplot(1, 2, i+1)
    plt.imshow(image, cmap='binary_r')
    plt.axis('off'), plt.title(titles[i])
plt.show()
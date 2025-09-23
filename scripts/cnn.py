import os
import time
from typing_extensions import Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# =========================
# HYPER PARAM
# =========================
BATCH_SIZE = 64
EPOCH_NUM = 50

# =========================
# 単純なCNN (入力は [B, C, H, W])
# =========================
class CNN(nn.Module):
    def __init__(self, n_out: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # [B,32,H/2,W/2]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [B,64,H/4,W/4]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# [B,128,H/8,W/8]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# [B,256,H/16,W/16]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))  # [B,256,1,1]
        )
        self.classifier = nn.Linear(256, n_out)

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, n_out]
        """
        feat = self.features(x).view(x.size(0), -1)   # [B,256]
        logits = self.classifier(feat)                # [B,n_out]
        return logits

# =========================
# 学習クラス（時系列なし）
# =========================
class deep_learning:
    def __init__(self, n_out: int = 2):
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device:", self.device)

        # model / opt / sched / loss
        self.net = CNN(n_out=n_out).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4, eps=1e-8, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=EPOCH_NUM, eta_min=1e-6)

        # クラス重み（2クラス想定：必要に応じて調整）
        balance_weights = torch.tensor([1.0, 2.1], device=self.device, dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=balance_weights)

        # 蓄積用（時系列なし：そのまま [N,C,H,W], [N] で貯める）
        self.first_flag = True
        self.loss_all = 0.0

        # ログ
        self.results_train = {'loss': [], 'accuracy': []}

        # 安全設定
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)

    # -------------------------
    # データ追加
    # img: [H, W, C] (np or torch) / dtypeはfloat32推奨（0-255なら後段で /255 してもOK）
    # label: int クラスID（CrossEntropyLoss 用）
    # -------------------------
    def make_dataset(self, img, label: int):
        """
        画像とラベルを一時リストにappendするだけ
        img: [H,W,C], label: int
        """
        x = torch.tensor(img, dtype=torch.float32)  # [H,W,C]
        if x.dim() != 3 or x.shape[2] != 3:
            raise ValueError("img must be [H,W,3]")

        # 必要なら正規化
        # if x.max() > 1.0:
        #     x = x / 255.0

        x = x.permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
        t = torch.tensor([int(label)], dtype=torch.long)  # [1]

        if not hasattr(self, "x_list"):
            self.x_list, self.t_list = [], []

        self.x_list.append(x)
        self.t_list.append(t)

        print(f"appended -> total {len(self.x_list)} samples (not yet concatenated)")
        return len(self.x_list)

    def finalize_dataset(self):
        """
        appendされたデータをまとめてcatしてTensorに変換
        """
        if not hasattr(self, "x_list") or len(self.x_list) == 0:
            raise RuntimeError("No data appended. Call make_dataset() first.")

        self.x_cat = torch.cat(self.x_list, dim=0)  # [N,C,H,W]
        self.t_cat = torch.cat(self.t_list, dim=0)  # [N]
        self.first_flag = False

        # メモリ解放
        del self.x_list
        del self.t_list

        print("Final dataset shapes -> X:", self.x_cat.shape, "Y:", self.t_cat.shape)
        return self.x_cat, self.t_cat

    # -------------------------
    # 学習
    # -------------------------
    def training(self):
        if not hasattr(self, 'x_cat') or not hasattr(self, 't_cat'):
            raise RuntimeError("No dataset yet. Call make_dataset() first.")

        dataset = TensorDataset(self.x_cat, self.t_cat)  # X:[N,C,H,W], Y:[N]
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)

        self.net.train()
        final_acc, final_loss = 0.0, 0.0

        for epoch in range(EPOCH_NUM):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for x_train, y_train in train_loader:
                x_train = x_train.to(self.device, non_blocking=True)  # [B,C,H,W]
                y_train = y_train.to(self.device, non_blocking=True)  # [B]

                self.optimizer.zero_grad()
                logits = self.net(x_train)                # [B,n_out]
                loss = self.criterion(logits, y_train)

                loss.backward()
                self.optimizer.step()

                # ログ
                epoch_loss += loss.item() * x_train.size(0)
                preds = torch.argmax(logits, dim=1)       # [B]
                correct += (preds == y_train).sum().item()
                total += y_train.size(0)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()

            avg_loss = epoch_loss / max(total, 1)
            acc = correct / max(total, 1)
            print(f'epoch [{epoch+1}/{EPOCH_NUM}] loss: {avg_loss:.4f} acc: {acc:.4f} lr: {current_lr:.6f}')

            self.results_train['loss'].append(avg_loss)
            self.results_train['accuracy'].append(acc)

            final_acc, final_loss = acc, avg_loss

        print("Finish learning")
        return final_acc, final_loss

    # -------------------------
    # 推論（単一画像）
    # img: [H,W,C]
    # returns: (pred_class:int, confidence:float, probs:Tensor[n_out])
    # -------------------------
    def test(self, img):
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(img, dtype=torch.float32)
            # x = x / 255.0  # 必要なら有効化
            x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)  # [1,C,H,W]
            logits = self.net(x)                 # [1,n_out]
            probs = F.softmax(logits, dim=1)[0]  # [n_out]
            conf, pred = torch.max(probs, dim=0)
            print("softmax:", probs.detach().cpu())
            print("confidence:", conf.item(), "predicted:", pred.item())
            return int(pred.item()), float(conf.item()), probs.detach().cpu()

    # -------------------------
    # データセット保存
    # -------------------------
    def save_tensor(self, path, file_name):
        os.makedirs(path, exist_ok=True)
        dataset = TensorDataset(self.x_cat, self.t_cat)
        torch.save(dataset, os.path.join(path, file_name))
        print("save_dataset_tensor")

    # -------------------------
    # モデル保存/読込
    # -------------------------
    def save(self, save_root):
        path = save_root + time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs(path, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(path, 'model.pt'))
        print("saved to:", path)

    def load(self, load_path):
        self.net.load_state_dict(torch.load(load_path, map_location=self.device))
        self.net.to(self.device)
        print("Loaded model from:", load_path)

if __name__ == '__main__':
    dl = deep_learning()
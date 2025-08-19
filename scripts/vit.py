from typing_extensions import Self
import numpy as np
import os
import time
from os.path import expanduser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, models
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

# test
from torcheval.metrics.functional import multiclass_accuracy

# HYPER PARAM
BATCH_SIZE = 16
FRAME_SIZE = 10
EPOCH_NUM = 10

class Net(nn.Module):
    """
    ViT (vit_b_16, ImageNet1K weights) を特徴抽出器として使用。
    ViTのheadsをIdentityに置き換えてCLSトークン(768次元)をそのまま取り出し、
    LSTM(入力768)→全結合 で分類します。
    """
    def __init__(self, n_out, num_frames_to_train_cnn=3):
        super().__init__()
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        # 分類ヘッドをアイデンティティに差し替えて768-d特徴を取得
        vit.heads = nn.Identity()
        self.vit = vit
        self.embed_dim = 768  # vit_b_16 の埋め込み次元

        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=512, num_layers=2, batch_first=True)
        self.output_layer = nn.Linear(512, n_out)
        self.num_frames_to_train_cnn = num_frames_to_train_cnn

    def forward(self, x):
        B, T, C, H, W = x.size()
        frame_features = []

        for t in range(T):
            frame = x[:, t, :, :, :]  # [B, C, H, W]
            if t < self.num_frames_to_train_cnn:
                feat = self.vit(frame)         # [B, 768]
            else:
                with torch.no_grad():
                    feat = self.vit(frame)     # [B, 768]
            frame_features.append(feat.unsqueeze(1))  # [B, 1, 768]

        frame_features = torch.cat(frame_features, dim=1)  # [B, T, 768]
        lstm_out, _ = self.lstm(frame_features)            # [B, T, 512]
        class_out = self.output_layer(lstm_out)            # [B, T, n_out]
        class_out = torch.mean(class_out, dim=1)           # [B, n_out]
        return class_out


class deep_learning:
    def __init__(self):
        # <tensor device choice>
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        self.net = Net(n_out=2)
        self.net.to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4, eps=1e-8, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=EPOCH_NUM, eta_min=1e-6)

        # ViT推奨の前処理（224入力, ImageNet統計）
        self.totensor = transforms.ToTensor()
        self.normalization = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_color = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
        self.random_erasing = transforms.RandomErasing(p=0.25, scale=(0.02, 0.09), ratio=(0.3, 3.3), value='random')

        self.count = 0
        self.accuracy = 0
        self.results_train = {'loss': [], 'accuracy': []}
        self.acc_list = []
        self.datas = []

        balance_weights = torch.tensor([1.0, 2.1]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=balance_weights)

        self.first_flag = True
        self.first_test_flag = True
        self.first_time_flag = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
        self.loss_all = 0.0
        self.intersection_test = torch.zeros(1, 8).to(self.device)
        self.old_label = [0, 0]
        self.diff_flag = False

    def make_dataset(self, img, intersection_label):
        # make tensor(T,C,H,W)
        if self.first_flag:
            self.x_cat = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            self.x_cat = self.x_cat.permute(0, 3, 1, 2)
            self.t_cat = torch.tensor([intersection_label], dtype=torch.float32)
            if self.first_time_flag:
                self.x_cat_time = torch.zeros(1, FRAME_SIZE, 3, 224, 224)
                self.t_cat_time = torch.clone(self.t_cat)
            self.first_flag = False
            self.first_time_flag = False

        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        t = torch.tensor([intersection_label], dtype=torch.float32)
        print(intersection_label)

        if intersection_label == self.old_label:
            self.diff_flag = False
            self.x_cat = torch.cat([self.x_cat, x], dim=0)
            print("cat x_cat", self.x_cat.shape)
        else:
            self.first_flag = True
            self.diff_flag = True
            print("change label")
        self.old_label = intersection_label

        if self.x_cat.size()[0] == FRAME_SIZE and self.diff_flag == False:
            print("make dataset")
            print("t_data:", t)
            self.x_cat_time = torch.cat((self.x_cat_time, self.x_cat.unsqueeze(0)), dim=0)
            self.t_cat_time = torch.cat((self.t_cat_time, t), dim=0)
            self.first_flag = True

        print("train x =", self.x_cat_time.shape, x.device, "train t = ", self.t_cat_time.shape, t.device)
        return self.x_cat_time, self.t_cat_time

    def training(self):
        self.device = torch.device('cuda')
        print(self.device)

        dataset = TensorDataset(self.x_cat_time, self.t_cat_time)
        train_dataset = DataLoader(dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu'),
                                   shuffle=True, pin_memory=True, num_workers=2)

        self.net.train()
        self.train_accuracy = 0

        for epoch in range(EPOCH_NUM):
            print('epoch', epoch)
            batch_loss = 0.0
            batch_accuracy = 0.0

            for x_train, t_label_train in train_dataset:
                x_train = x_train.to(self.device, non_blocking=True)
                t_label_train = t_label_train.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                y_train = self.net(x_train)
                loss_all = self.criterion(y_train, t_label_train)

                self.train_accuracy += torch.sum(
                    torch.max(y_train, 1)[1] == torch.max(t_label_train, 1)[1]
                ).item()

                print("epoch:", epoch, "accuracy :", self.train_accuracy, "/", len(t_label_train),
                      (self.train_accuracy / len(t_label_train)) * 100, "%", "loss :", loss_all.item())

                loss_all.backward()
                self.optimizer.step()

                self.loss_all = loss_all.item()
                batch_loss += self.loss_all
                batch_accuracy += multiclass_accuracy(
                    input=torch.max(y_train, 1)[1],
                    target=torch.max(t_label_train, 1)[1],
                    num_classes=8,
                    average="micro"
                ).item()

                self.count += 1
                self.train_accuracy = 0

            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            print(f'epoch [{epoch+1}/{EPOCH_NUM}], loss: {batch_loss/len(train_dataset):.4f}, '
                  f'accuracy: {batch_accuracy/len(train_dataset)}, lr: {current_lr:.6f}')

        print("Finish learning")
        finish_flag = True
        return self.train_accuracy, self.loss_all

    def test(self, img):
        self.net.eval()
        if self.first_test_flag:
            self.x_cat_test = torch.tensor(img, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.x_cat_test = self.x_cat_test.permute(0, 3, 1, 2)
            self.first_test_flag = False
            self.prev_prediction = 0

        x = torch.tensor(img, dtype=torch.float32, device=self.device).unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        self.x_cat_test = torch.cat([self.x_cat_test, x], dim=0)

        if self.x_cat_test.size()[0] == FRAME_SIZE:
            with torch.no_grad():
                logits = self.net(self.x_cat_test.unsqueeze(0))   # [1, n_out]
                probs = F.softmax(logits, dim=1).squeeze(0)       # [n_out]
                confidence, predicted = torch.max(probs, dim=0)
                print("softmax output:", probs)
                print("confidence:", confidence.item(), "predicted:", predicted.item())

                self.x_cat_test = self.x_cat_test[1:]

                if confidence.item() >= 0.7:
                    self.prev_prediction = predicted.item()
                print("output:", self.prev_prediction)

        return self.prev_prediction

    def save_tensor(self, path, file_name):
        os.makedirs(path)
        dataset = TensorDataset(self.x_cat_time, self.t_cat_time)
        torch.save(dataset, path + file_name)
        print("save_dataset_tensor:")

    def save(self, save_path):
        path = save_path + time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs(path, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(path, 'model.pt'))

    def load(self, load_path):
        self.net.load_state_dict(torch.load(load_path))
        print("Loaded model from:", load_path)

if __name__ == '__main__':
    dl = deep_learning()

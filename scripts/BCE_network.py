from typing_extensions import Self
import numpy as np
import matplotlib as plt
import os
import time
from os.path import expanduser

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from torchvision import transforms, models
import torch.optim as optim
import random
import glob
from torcheval.metrics.functional import binary_accuracy

from torchvision.utils import save_image  # ← これが必要

# HYPER PARAM
BATCH_SIZE = 32
FRAME_SIZE = 16
EPOCH_NUM = 10

class Net(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        v3 = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        v3.classifier[-1] = nn.Linear(in_features=1280, out_features=1280)
        self.v3_layer = v3
        self.lstm = nn.LSTM(input_size=1280, hidden_size=512, num_layers=2, batch_first=True)
        self.output_layer = nn.Linear(512, n_out)

    def forward(self, x):
        frameFeatures = torch.empty(size=(x.size()[0], x.size()[1], 1280), device=x.device)
        for t in range(0, x.size()[1]):
            frame = x[:, t, :, :, :]
            frame_feature = self.v3_layer(frame)
            frameFeatures[:, t, :] = frame_feature
        lstm_out, _ = self.lstm(frameFeatures)
        class_out = self.output_layer(lstm_out)
        class_out = torch.sigmoid(class_out)
        class_out = torch.mean(class_out, dim=1)
        return class_out

class deep_learning:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.net = Net(n_out=1)
        self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), eps=1e-2, weight_decay=5e-4)
        self.totensor = transforms.ToTensor()
        self.normalization = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform_color = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
        self.random_erasing = transforms.RandomErasing(p=0.25, scale=(0.02, 0.09), ratio=(0.3, 3.3), value='random')
        self.criterion = nn.BCEWithLogitsLoss()
        self.count = 0
        self.results_train = {'loss': [], 'accuracy': []}
        self.first_flag = True
        self.first_time_flag = True
        self.first_test_flag = True
        self.loss_all = 0.0
        self.old_label = 0
        self.diff_flag = False

    def make_dataset(self, img, intersection_label):
        if self.first_flag:
            self.x_cat = torch.tensor(img, dtype=torch.float32, device=self.device).unsqueeze(0).permute(0, 3, 1, 2)
            self.t_cat = torch.tensor([float(intersection_label)], dtype=torch.float32, device=self.device)
            if self.first_time_flag:
                self.x_cat_time = torch.zeros(1, FRAME_SIZE, 3, 48, 64).to(self.device)
                self.t_cat_time = torch.clone(self.t_cat)
            self.first_flag = False
            self.first_time_flag = False

        x = torch.tensor(img, dtype=torch.float32, device=self.device).unsqueeze(0).permute(0, 3, 1, 2)
        t = torch.tensor([float(intersection_label)], dtype=torch.float32, device=self.device)
        if intersection_label == self.old_label:
            self.diff_flag = False
            self.x_cat = torch.cat([self.x_cat, x], dim=0)
        else:
            self.first_flag = True
            self.diff_flag = True
            print("change label")
        self.old_label = intersection_label

        if self.x_cat.size()[0] == FRAME_SIZE and not self.diff_flag:
            self.x_cat_time = torch.cat((self.x_cat_time, self.x_cat.unsqueeze(0)), dim=0)
            self.t_cat_time = torch.cat((self.t_cat_time, t), dim=0)
            self.first_flag = True

        print("train x =",self.x_cat_time.shape, "train t = " ,self.t_cat_time.shape)

        return self.x_cat_time, self.t_cat_time

    def training(self, load_x_tensor, load_t_tensor, load_flag):
        self.device = torch.device('cuda')
        if load_flag:
            load_x_tensor = torch.load(load_x_tensor)
            load_t_tensor = torch.load(load_t_tensor)

        # print("x_tensor", load_x_tensor)
        print("t_tensor", load_t_tensor.shape)
        dataset = TensorDataset(load_x_tensor, load_t_tensor)
        train_dataset = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.net.train()

        for epoch in range(EPOCH_NUM):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            for x_train, t_label_train in train_dataset:
                x_train = x_train.to(self.device)
                t_label_train = t_label_train.unsqueeze(1)
                self.optimizer.zero_grad()
                y_train = self.net(x_train)
                loss = self.criterion(y_train, t_label_train)
                loss.backward()
                self.optimizer.step()

                preds = (torch.sigmoid(y_train) > 0.5).float()
                acc = (preds == t_label_train).float().mean().item()

                epoch_loss += loss.item()
                epoch_accuracy += acc

            epoch_loss /= len(train_dataset)
            epoch_accuracy /= len(train_dataset)
            print(f"Epoch {epoch} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        return epoch_accuracy, epoch_loss

    def test(self, img):
        self.net.eval()
        result = None
        if self.first_test_flag:
            self.x_cat_test = torch.tensor(img, dtype=torch.float32, device=self.device).unsqueeze(0).permute(0, 3, 1, 2)
            self.first_test_flag = False
        x = torch.tensor(img, dtype=torch.float32, device=self.device).unsqueeze(0).permute(0, 3, 1, 2)
        self.x_cat_test = torch.cat([self.x_cat_test, x], dim=0)
        # print("x_test_cat:",self.x_cat_test.shape)
        if self.x_cat_test.size()[0] == FRAME_SIZE:
            out = self.net(self.x_cat_test.unsqueeze(0))
            # print("out", out)
            self.x_cat_test = self.x_cat_test[1:]
            # out = torch.sigmoid(out)
            print("result", out)
            if out.item() > 0.5:
                result = 1
            else:
                result = 0
        return result

    def save_tensor(self, input_tensor, save_path, file_name):
        path = save_path + time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs(path)
        torch.save(input_tensor, path + file_name)
        print("save_dataset_tensor:",)

    def save(self, save_path):
        path = save_path + time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs(path, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(path, 'model.pt'))

    def load(self, load_path):
        self.net.load_state_dict(torch.load(load_path))
        print("Loaded model from:", load_path)

if __name__ == '__main__':
    dl = deep_learning()

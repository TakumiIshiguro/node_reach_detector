from typing_extensions import Self
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from os.path import expanduser


import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset,Subset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random
import glob
#test
from torcheval.metrics.functional import multiclass_accuracy

# HYPER PARAM
BATCH_SIZE = 32
FRAME_SIZE = 16
EPOCH = 200
CLIP_VALUE = 4.0

class Net(nn.Module):
    def __init__(self, n_out):
        super().__init__()
    # <Network mobilenetv2>
        v3 = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        v3.classifier[-1] = nn.Linear(in_features=1280, out_features=1280)
    #<LSTM + OUTPUT>
        self.lstm = nn.LSTM(input_size=1280,
                            hidden_size=512, num_layers=2, batch_first=True)
        self.output_layer = nn.Linear(512, n_out)
    # <CNN layer>
        self.v3_layer = v3

    def forward(self, x):
         # x's dimension: [B, T, C, H, W]

        # frameFeatures' dimension: [B, T, CNN's output dimension(1280)]
        frameFeatures = torch.empty(size=(x.size()[0], x.size()[1], 1280), device='cuda')
        for t in range(0, x.size()[1]):
            #<x[B,T,C,H,W]->CNN[B,T,1280]>
            # print("forword_x:",x.shape)
            frame = x[:,t, :, :,:]
            # print("forword_frame:",frame.shape)
            frame_feature = self.v3_layer(frame)
            # print(frame_feature.shape)
            #[B,seq_len,H]
            frameFeatures[:,t, :] = frame_feature
        #<CNN[B,T,1280] -> lstm[B,1280,512]>
        # print("lstm_in:",frameFeatures.shape)
        lstm_out, _ = self.lstm(frameFeatures)
        #<lstm[B,1280]-> FC[B,4,512]>
        # print("lstm_out:",lstm_out.shape)
        class_out = self.output_layer(lstm_out)
        # print("class_out",class_out.shape)
        class_out = torch.mean(class_out, dim=1)
        # print("class_out_mean",class_out.shape)
        return class_out
    
class deep_learning:
    def __init__(self):
        # <tensor device choice>
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.net = Net(n_out=2).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), eps=1e-2, weight_decay=5e-4)
        self.totensor = transforms.ToTensor()
        # self.transform_train = transforms.Compose([transforms.RandomRotation(10),
        #                                            transforms.ColorJitter(brightness=0.3, saturation=0.3)])
        # self.normalization = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # self.transform_color = transforms.ColorJitter(
        #     brightness=0.5, contrast=0.5, saturation=0.5)
        # self.random_erasing = transforms.RandomErasing(
        #     p=0.25, scale=(0.02, 0.09), ratio=(0.3, 3.3), value= 'random')
        self.count = 0
        self.accuracy = 0
        self.results_train = {}
        self.results_train['loss'], self.results_train['accuracy'] = [], []
        self.acc_list = []
        self.datas = []
        self.first_flag = True
        self.first_test_flag = True
        self.first_time_flag = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
        self.loss_all = 0.0
        self.intersection_test = torch.zeros(1,8).to(self.device)
        self.old_label = 0
        self.diff_flag = False

    def call_dataset(self):
        if self.first_flag:
            return
        dataset = TensorDataset(self.img_time, self.node_time)
        return self.img, self.node
    
    def load_dataset(self, image_path, node_path):
        self.img_time = torch.load(image_path)
        self.node_time = torch.load(node_path)

        # データ数を確認
        total_samples = self.img_time.shape[0]
        print(f"Total samples: {total_samples}")

        # CUDA上にある場合はCPUへ移動
        if self.node_time.is_cuda:
            node_labels = self.node_time.cpu().numpy().squeeze().astype(int)
        else:
            node_labels = self.node_time.numpy().squeeze().astype(int)

        # ラベルごとの件数を表示
        unique_labels, counts = np.unique(node_labels, return_counts=True)
        self.num_classes = int(np.max(unique_labels)) + 1  # ← クラス数推定

        print("Samples per node label:")
        for label, count in zip(unique_labels, counts):
            print(f"  Node {label}: {count} samples")

        self.create_weighted_loss(self.node_time)

        return self.img_time, self.node_time

    
    def plot_node_distribution(self, labels):

        if labels.is_cuda:
            labels = labels.cpu()
        labels = labels.numpy().squeeze().astype(int)
        unique_labels, counts = np.unique(labels, return_counts=True)

        plt.figure(figsize=(10, 6))
        plt.bar(unique_labels, counts, edgecolor='black')
        plt.xlabel("Node ID")
        plt.ylabel("Number of Images")
        plt.title("Distribution of Images per Node")
        plt.grid(axis='y')
        plt.xticks(unique_labels)  # ノード番号をx軸に明示
        plt.show()


    def create_weighted_loss(self, node_tensor):
        if node_tensor.is_cuda:
            node_tensor = node_tensor.cpu()
        labels = node_tensor.numpy().squeeze().astype(int)

        # クラス数を自動検出してクラスごとの出現回数を数える
        unique_classes, class_counts = np.unique(labels, return_counts=True)
        max_count = np.max(class_counts)

        class_weights = max_count / class_counts

        num_classes = int(np.max(unique_classes)) + 1
        full_weights = np.ones(num_classes, dtype=np.float32)
        for cls, w in zip(unique_classes, class_weights):
            full_weights[cls] = w
        weight_tensor = torch.tensor(full_weights, dtype=torch.float32).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        # self.criterion = nn.CrossEntropyLoss()

        print("Class counts:", dict(zip(unique_classes, class_counts)))
        print("Class weights:", weight_tensor)

    def make_dataset(self, img, node_num):
        # make tensor(T,C,H,W)
        if self.first_flag:
            self.img = torch.tensor(
                img, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.img = self.img.permute(0, 3, 1, 2)
            self.node = torch.tensor(
                [node_num], dtype=torch.long, device=self.device)
            if self.first_time_flag:
                self.img_time = torch.zeros(1, FRAME_SIZE, 3, 48, 64).to(self.device) 
                self.node_time = torch.clone(self.node)

            self.first_flag = False
            self.first_time_flag = False
        # <to tensor img(x),node_num(t)>
        x = torch.tensor(img, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        # <(T,H,W,Channel) -> (T ,Channel, H,W)>
        x = x.permute(0, 3, 1, 2)
        # <(t dim [4]) -> [1,4] >
        node = torch.tensor([node_num], dtype=torch.long,
                         device=self.device)

        if node_num == self.old_label:
            self.diff_flag = False
            self.img = torch.cat([self.img, x], dim=0)
            print("cat img",self.img.shape)
        else:
            self.first_flag = True
            self.diff_flag = True
            print("change label")
        # <self.img (B,C,H,W) = (8,3,48,64))>
        self.old_label = node_num
       
        #<add tensor(B,C,H,W) to dateset_tensor(B,T,C,H,W)>
        if self.img.size()[0] == FRAME_SIZE and self.diff_flag ==False:
            # <self.img_time (B,T,C,H,W) = (8,8,3,48,64))>
            print("make dataset")
            print("node_data:", node)
            #print("img_time:",self.img_time.shape,"img_sq:",self.img.unsqueeze(0).shape)
            self.img_time = torch.cat((self.img_time, self.img.unsqueeze(0)), dim=0)
            #<self.node_time (B,T,Size) = (8,8,4)>
            self.node_time = torch.cat((self.node_time, node), dim=0)
            self.first_flag = True
    # <make dataset>
        print("train img =", self.img_time.shape, x.device, "train node = ", self.node_time.shape, node.device)

        return self.img_time, self.node_time
        
    def trains(self, img, node_num):
        # self.device = torch.device('cuda')
        print(self.device)
        dataset = TensorDataset(img, node_num)
        train_dataset = DataLoader(dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu'), shuffle=True)
    # <training mode>
        self.train_accuracy = 0
    
    # <split dataset and to device>
        for epoch in range(EPOCH):
            self.net.train()
            loss_all = 0.0
            correct = 0
            total = 0
            count = 0
            
            for img, node_train in train_dataset:
                img = img.to(self.device, non_blocking=True)
                node_train = node_train.to(self.device, non_blocking=True)
                # print(node_train.dtype, node_train.shape)
                # print(node_train[:5])

        # <use transform>

        # <learning>
                self.optimizer.zero_grad()
                y_train = self.net(img)
                # print("y = ",y_train,"t=",node_train)
                loss = self.criterion(y_train, node_train)
                loss.backward()
                self.optimizer.step()
                loss_all += loss.item()

                # print(y_train)
                predicted = torch.argmax(y_train, dim=1)
                correct += (predicted == node_train).sum().item()
                total += node_train.size(0)

                count += 1

            average_loss = loss_all / count
            accuracy = correct / total if total > 0 else 0.0
            print(f"Epoch {epoch+1}, Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

        print("Finish learning!!")
        finish_flag = True

    def test(self, img):
        self.net.eval()
    # <to tensor img(x)>
        if self.first_test_flag:
            self.img_test = torch.tensor(
                img, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.img_test = self.img_test.permute(0, 3, 1, 2)
            self.first_test_flag = False
        
        x = torch.tensor(
            img, dtype=torch.float32, device=self.device).unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        self.img_test = torch.cat([self.img_test, x], dim=0)
        # print("x_tesnode:",self.img_test.shape)
        if self.img_test.size()[0] == FRAME_SIZE:
            self.intersection_test = self.net(self.img_test.unsqueeze(0))
            print("s:",self.intersection_test.shape)
            self.img_test = self.img_test[1:]
        # print(x_test_ten.shape,x_test_ten.device,c_test.shape,c_test.device)
    # <test phase>        
        return torch.max(self.intersection_test, 1)[1].item()

    def save_tensor(self, dataset_tensor, save_path, file_name):
        os.makedirs(save_path)
        torch.save(dataset_tensor, save_path+file_name)
        print("save_path: ",save_path + file_name)

    def save(self, save_path):
        # <model save>
        path = save_path + time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs(path)
        torch.save(self.net.state_dict(), path + '/model.pt')
    
    def load(self, load_path):
        # <model load>
        self.net.load_state_dict(torch.load(load_path))
        print(load_path)

if __name__ == '__main__':
    dl = deep_learning()

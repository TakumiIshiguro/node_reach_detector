#!/usr/bin/env python3
from __future__ import print_function

from numpy import dtype
import roslib
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from network import *
from skimage.transform import resize
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from nav_msgs.msg import Path
from std_msgs.msg import Int8MultiArray
from scenario_navigation_msgs.msg import cmd_dir_intersection
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Empty
from std_srvs.srv import SetBool, SetBoolResponse
import csv
import os
import time
import copy
import sys
import tf
from nav_msgs.msg import Odometry

class node_reach_detector:
    def __init__(self):
        rospy.init_node('node_reach_detector', anonymous=True)

        self.dl = deep_learning()
        self.name = 'test'
        self.save_base = roslib.packages.get_pkg_dir('node_reach_detector') + '/data/'
        self.save_dataset = self.save_base + '/dataset/'
        self.save_model = self.save_base + '/model/'
        self.load_base = roslib.packages.get_pkg_dir('dataset_creator') + '/dataset/' + str(self.name)

        self.load_dataset = os.path.join(self.save_dataset, 'learning/dataset.pt')
        self.load_test_dataset = os.path.join(self.save_dataset, 'test/dataset.pt')

        self.image_dirs = {
            'center': os.path.join(self.load_base, 'image/center'),
            'left':   os.path.join(self.load_base, 'image/left'),
            'right':  os.path.join(self.load_base, 'image/right'),
            'resize':  os.path.join(self.load_base, 'image/resize')
        }

        self.inter_csv = os.path.join(self.load_base, 'inter.csv')

        print("[INFO] Dataset load path:", self.load_base)

    def load_images(self, path):
        image_data = {}
        files = sorted(
            [f for f in os.listdir(path) if f.endswith('.png')],
            key=lambda f: int(os.path.splitext(f)[0])
        )
        for file in files:
            try:
                episode = int(os.path.splitext(file)[0])
                img = cv2.imread(os.path.join(path, file))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGBに変換
                img = img.astype(np.float32) / 255.0        # 0〜1に正規化
                image_data[episode] = img
                print(f"[INFO] Loaded image for episode {episode}")
            except Exception as e:
                print(f"[WARN] Failed to load image {file}: {e}")
        return image_data
    
    def load_inter_csv(self, path):
        data = {}
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                try:
                    episode = int(row[0])
                    inter_tuple = eval(row[1].strip())
                    data[episode] = list(inter_tuple)
                except Exception as e:
                    print(f"[WARN] Failed to parse vel row {row}: {e}")
        return data
    
    def main(self):
        # print("[INFO] Loading data...")
        # inter_dict = self.load_inter_csv(self.inter_csv)
        # for view in ['resize']:
        #     img_dict = self.load_images(self.image_dirs[view])
        #     print(f"[INFO] Loaded {len(img_dict)} images for view: {view}")

        #     episodes = sorted(set(img_dict.keys()) & set(inter_dict.keys()))
        #     for ep in episodes:
        #         img = img_dict[ep]
        #         # cv2.imshow("center", img)
        #         # cv2.waitKey(1)
        #         inter_flg = inter_dict[ep]  

        #         self.dl.make_dataset(img, inter_flg)

        # self.dl.save_tensor(dataset, self.save_dataset, '/dataset.pt')
        dataset = self.dl.load_tensor(self.load_dataset)
        test_dataset = self.dl.load_tensor(self.load_test_dataset)
        self.dl.training(dataset, test_dataset)

        self.dl.save(self.save_model)
        print("[INFO] Training complete. Model saved to:", self.save_model)

        os.system('killall roslaunch')
        sys.exit()

if __name__ == '__main__':
    rg = node_reach_detector()
    rg.main()
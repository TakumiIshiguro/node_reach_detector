#!/usr/bin/env python3

import numpy as np
import roslib
roslib.load_manifest('node_reach_detector')
import rospy
from cnn import *
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from skimage.transform import resize
import os
import sys
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import csv
import time
from sensor_msgs.msg import Joy
import copy
import yaml

# todo:delete
from scenario_navigation_msgs.msg import cmd_dir_intersection
from std_srvs.srv import SetBool, SetBoolResponse

class node_reach_detector:
    def __init__(self):
        rospy.init_node('node_reach_detector', anonymous=True)
        self.num = int(rospy.get_param("/node_reach_detector/num", "1"))
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera_center/usb_cam/image_raw", Image, self.callback)

        self.dl = deep_learning()
        self.action = 0.0
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)
    
        self.inter_flg = False
        self.ignore_flg = False    
        self.loop_srv = rospy.Service('/loop_count', SetBool, self.callback_loop_count)
        self.loop_count_flag = False

        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.path = roslib.packages.get_pkg_dir('node_reach_detector') + '/data/dataset/' + str(self.start_time)

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_loop_count(self, data):
        self.loop_count_flag = data.data

    def joy_callback(self, data):
        if data.buttons[1] == 1:
            self.inter_flg = True
        else: 
            self.inter_flg = False
            
        if data.buttons[2] == 1:
            self.ignore_flg = True
        else:
            self.ignore_flg = False

    # def preprocess_for_mobilenet(self, image):
    #     """
    #     MobileNetV3用に画像を前処理：
    #     - BGR → RGB
    #     - 正方形中央クロップ → 224x224へリサイズ
    #     - 0〜1のfloat32に変換
    #     """
    #     # BGR → RGB
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     h, w = image.shape[:2]
    #     crop_size = min(h, w)
    #     left = (w - crop_size) // 2
    #     top = (h - crop_size) // 2
    #     image_crop = image[top:top+crop_size, left:left+crop_size]

    #     # リサイズ & 正規化
    #     image_resized = resize(image_crop, (224, 224), mode='constant')
    #     return image_resized

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            print("No Image")
            return
        
        # crooped_img = self.cv_image[:, 80:560]
        # crooped_left_img = self.cv_left_image[156:, :]
        # crooped_right_img = self.cv_right_image[156:, :]
        # img = resize(crooped_img, (227, 227), mode='constant')
        # img_left = resize(self.cv_left_image, (48, 64), mode='constant')
        # img_right = resize(self.cv_right_image, (48, 64), mode='constant')

        img = resize(self.cv_image, (224, 224), mode='constant')
        cv2.imshow("resize", img)
        cv2.imshow("center", self.cv_image)
        cv2.waitKey(1)

        if self.inter_flg:
            self.dl.make_dataset(img, 1)
            print("label 1")
        elif self.ignore_flg:
            pass
        else:
            self.dl.make_dataset(img, 0)
            print("label 0")

        if self.loop_count_flag:
            img_tensor, node_tensor = self.dl.finalize_dataset() 
            self.dl.save_tensor(self.path, 'dataset.pt')
            self.dl.training()
            self.loop_count_flag = False
            os.system('killall roslaunch')
            sys.exit()
        else :
            pass
      
if __name__ == '__main__':
    rg = node_reach_detector()
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
#!/usr/bin/env python3

import numpy as np
import roslib
roslib.load_manifest('node_reach_detector')
import rospy
from network import *
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


# def load_config(filename="config.yaml"):
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     config_path = os.path.join(script_dir, "..", "config", filename)  # configディレクトリ内を想定
#     with open(config_path, 'r') as file:
#         return yaml.safe_load(file)
    
# config = load_config()

class node_reach_detector:
    def __init__(self):
        rospy.init_node('node_reach_detector', anonymous=True)
        self.num = int(rospy.get_param("/node_reach_detector/num", "1"))
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera_center/image_raw", Image, self.callback)

        self.dl = deep_learning()
        self.action = 0.0
        self.cv_image = np.zeros((480,640,3), np.uint8)

        self.tracker_sub = rospy.Subscriber("/tracker", Odometry, self.callback_tracker)
        self.cmd_dir = (1, 0, 0)
        self.old_cmd_dir = (1, 0, 0)
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)
        self.joy_flg = False
        self.inter_flg = False
        
        self.loop_srv = rospy.Service('/loop_count', SetBool, self.callback_loop_count)
        self.loop_count_flag = False

        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.save_image_path = roslib.packages.get_pkg_dir('node_reach_detector') + '/data/dataset/' + str(self.start_time) + '/image/'
        self.save_node_path = roslib.packages.get_pkg_dir('node_reach_detector') + '/data/dataset/' + str(self.start_time) + '/node/'

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)

    def callback_loop_count(self, data):
        self.loop_count_flag = data.data

    def joy_callback(self, data):
        # buttons[1] が押されているかチェック
        if data.buttons[1] == 1:
            self.joy_flg = True

        if data.buttons[6] == 1:
            self.cmd_dir = (0, 1, 0)
        elif data.buttons[7] == 1:
            self.cmd_dir = (0, 0, 1)
        else:
            self.cmd_dir = (1, 0, 0)

        if data.buttons[5] == 1:
            self.inter_flg = True
        else: 
            self.inter_flg = False

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            print("No Image")
            return
        if self.cmd_dir == (0, 0, 0):
            print("No direction")
            return

        if self.old_cmd_dir != self.cmd_dir and self.cmd_dir != (1, 0, 0):
            pass
        
        # crooped_img = self.cv_image[:, 80:560]
        # crooped_left_img = self.cv_left_image[156:, :]
        # crooped_right_img = self.cv_right_image[156:, :]
        # img = resize(crooped_img, (227, 227), mode='constant')
        # img_left = resize(self.cv_left_image, (48, 64), mode='constant')
        # img_right = resize(self.cv_right_image, (48, 64), mode='constant')


        img = resize(self.cv_image, (48, 64), mode='constant')

        if self.cmd_dir == (0, 1, 0) or self.cmd_dir == (0, 0, 1) or self.inter_flg:
            img_tensor, node_tensor = self.dl.make_dataset(img, (0, 1))
            print("label 0")
        else:
            img_tensor, node_tensor = self.dl.make_dataset(img, (1, 0))
            print("label 1")

        if self.joy_flg: 
            self.dl.save_tensor(img_tensor, self.save_image_path, '/image.pt')
            self.dl.save_tensor(node_tensor, self.save_node_path, '/node.pt')
            os.system('killall roslaunch')
            sys.exit()

        if self.loop_count_flag:
            self.dl.save_tensor(img_tensor, self.save_image_path,'/image.pt')
            self.dl.save_tensor(node_tensor, self.save_node_path, '/node.pt')
            self.loop_count_flag = False
            os.system('killall roslaunch')
            sys.exit()
        else :
            pass
      
if __name__ == '__main__':
    rg = node_reach_detector()
    r = rospy.Rate(8.0)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
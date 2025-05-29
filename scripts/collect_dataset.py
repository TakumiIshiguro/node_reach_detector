#!/usr/bin/env python3

import numpy as np
import roslib
roslib.load_manifest('node_reach_detector')
import rospy
from test_network import *
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
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)
        self.image_left_sub = rospy.Subscriber("/camera_left/image_raw", Image, self.callback_left_camera)
        self.image_right_sub = rospy.Subscriber("/camera_right/image_raw", Image, self.callback_right_camera)
        self.vel = Twist()
        self.vel_sub = rospy.Subscriber("/joy_vel", Twist, self.callback_vel)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.dl = deep_learning()
        self.action = 0.0
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.cv_left_image = np.zeros((480,640,3), np.uint8)
        self.cv_right_image = np.zeros((480,640,3), np.uint8)
        self.tracker_sub = rospy.Subscriber("/tracker", Odometry, self.callback_tracker)
        self.cmd_dir = (1, 0, 0)
        self.old_cmd_dir = (1, 0, 0)
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)
        self.joy_flg = False
        self.ignore_flg = False

        # todo: delete
        self.cmd_dir_sub = rospy.Subscriber("/cmd_dir_intersection", cmd_dir_intersection, self.callback_cmd,queue_size=1)
        self.cmd_dir_data = [0,0,0,0,0,0,0,0]
        
        self.loop_srv = rospy.Service('/loop_count', SetBool, self.callback_loop_count)
        self.loop_count_flag = False
        #

        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.save_image_path = roslib.packages.get_pkg_dir('node_reach_detector') + '/data/dataset/' + str(self.start_time) + '/image/'
        self.save_node_path = roslib.packages.get_pkg_dir('node_reach_detector') + '/data/dataset/' + str(self.start_time) + '/node/'

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_left_camera(self, data):
        try:
            self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_right_camera(self, data):
        try:
            self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel.angular.z

    def callback_tracker(self, data):
        self.pos_x = data.pose.pose.position.x
        self.pos_y = data.pose.pose.position.y

#
    def callback_cmd(self, data):
        self.cmd_dir_data = data.intersection_label

    def callback_loop_count(self, data):
        resp = SetBoolResponse()
        self.loop_count_flag = data.data
        resp.message = "Training: " + str(self.learning)
        resp.success = True
        return resp
#

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
            self.ignore_flg = True
        else: 
            self.ignore_flg = False

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.cv_left_image.size != 640 * 480 * 3:
            return
        if self.cv_right_image.size != 640 * 480 * 3:
            return
        if self.cmd_dir == (0, 0, 0):
            return

        if self.ignore_flg:
            pass
        else:
            if self.old_cmd_dir != self.cmd_dir and self.cmd_dir != (1, 0, 0):
            #     self.node_num += 1
            # self.old_cmd_dir = self.cmd_dir
                pass
            
            # crooped_img = self.cv_image[:, 80:560]
            # crooped_left_img = self.cv_left_image[156:, :]
            # crooped_right_img = self.cv_right_image[156:, :]
            # img = resize(crooped_img, (227, 227), mode='constant')
            # img_left = resize(self.cv_left_image, (48, 64), mode='constant')
            # img_right = resize(self.cv_right_image, (48, 64), mode='constant')


            img = resize(self.cv_image, (48, 64), mode='constant')
            img_left = resize(self.cv_left_image, (48, 64), mode='constant')
            img_right = resize(self.cv_right_image, (48, 64), mode='constant')
            print("cmd_dir_data: ", self.cmd_dir_data)

            # if self.cmd_dir == (0, 1, 0) or self.cmd_dir == (0, 0, 1):
            if  self.cmd_dir_data == (1,0,0,0,0,0,0,0):
                self.img_tensor, self.node_tensor = self.dl.make_dataset(img, (1, 0))
                print("label 0")
                # self.dl.make_dataset(img_left, self.node_num)
                # self.dl.make_dataset(img_right, self.node_num)
            else:
                self.img_tensor, self.node_tensor = self.dl.make_dataset(img, (0, 1))
                print("label 1")
                # self.dl.make_dataset(img_left, 0)
                # self.dl.make_dataset(img_right, 0)

            if self.joy_flg: 
                # img, node_num = self.dl.call_dataset()
                self.dl.save_tensor(self.img_tensor, self.save_image_path, '/image.pt')
                self.dl.save_tensor(self.node_tensor, self.save_node_path, '/node.pt')
                os.system('killall roslaunch')
                sys.exit()

            if self.loop_count_flag:
                self.dl.save_tensor(self.img_tensor, self.save_image_path,'/image.pt')
                self.dl.save_tensor(self.node_tensor, self.save_node_path, '/node.pt')
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
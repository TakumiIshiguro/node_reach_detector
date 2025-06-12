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
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera_center/image_raw", Image, self.callback)
        self.vel = Twist()
        self.dl = deep_learning()
        self.episode = 0
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.load_path =roslib.packages.get_pkg_dir('node_reach_detector') + '/data/model/test/model.pt'
        self.first_flag = False
        # todo: delete        
        self.intersection = ["straight_road", "intersection"]

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
        img = resize(self.cv_image, (48, 64), mode='constant')
        ros_time = str(rospy.Time.now())

        if self.episode == 0:
            self.dl.load(self.load_path)
            print("load model: ",self.load_path)
        
        intersection = self.dl.test(img)
        # self.intersection.intersection_name = self.intersection_list[intersection]
        print(intersection)
        # self.intersection_pub.publish(self.intersection)
        # print("test" + str(self.episode) +", intersection_name: " + str(self.intersection.intersection_name))

        self.episode += 1

      
if __name__ == '__main__':
    rg = node_reach_detector()
    r = rospy.Rate(8.0)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
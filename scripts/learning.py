#!/usr/bin/env python3
from __future__ import print_function

from numpy import dtype
import roslib
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from test_network import *
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
        self.learning = True
        self.mode_save_srv = rospy.Service('/model_save', Trigger, self.callback_model_save)
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.name = '2label'
        self.save_path = roslib.packages.get_pkg_dir('node_reach_detector') + '/data/model/' + str(self.name) + '/'
        # self.load_path =roslib.packages.get_pkg_dir('node_reach_detector') + '/data/model/cit3f/direction/1/model.pt'
        self.load_image_path = roslib.packages.get_pkg_dir('node_reach_detector') + '/data/dataset/' + str(self.name) + '/image/2label/' + '/image.pt'
        self.load_node_path = roslib.packages.get_pkg_dir('node_reach_detector') + '/data/dataset/' + str(self.name) + '/node/2label/' + '/node.pt'
        self.start_time_s = rospy.get_time()

    def callback_model_save(self, data):
        model_res = SetBoolResponse()
        self.dl.save(self.save_path)
        model_res.message ="model_save"
        model_res.success = True
        return model_res
    
    def loop(self):
        self.dl.cat_training(self.load_image_path, self.load_node_path, True)
            # test_dataset = self.dl.load_dataset(self.test_image_path, self.test_dir_path, self.test_vel_path)
        # a = self.dl.plot_node_distribution(node)
        # self.dl.trains(img, node)
        # self.dl.save(self.save_path)
        print("Finish learning")
        os.system('killall roslaunch')
        sys.exit()

if __name__ == '__main__':
    rg = node_reach_detector()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
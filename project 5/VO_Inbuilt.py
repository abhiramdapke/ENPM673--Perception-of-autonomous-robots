# import sys,os

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import pprint
from scipy.optimize import least_squares
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def getCameraMatrix(path):
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel(path)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K, LUT


def undistortImageToGray(img, LUT):
    colorimage = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    undistortedimage = UndistortImage(colorimage, LUT)
    gray = cv2.cvtColor(undistortedimage, cv2.COLOR_BGR2GRAY)
    return gray


def features(img1, img2):
    orb = cv2.ORB_create(2000)

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # Brute force matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    # Initialize lists
    list_kp1 = []
    list_kp2 = []

    # For each match...
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Append to each list
        list_kp1.append((x1, y1))
        list_kp2.append((x2, y2))

    list_kp1 = np.array(list_kp1)
    list_kp2 = np.array(list_kp2)
    return list_kp1, list_kp2


def main():
    BasePath = './stereo/centre/'
    K, LUT = getCameraMatrix('./model')
    images = []
    H1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    cam_pos = np.array([0,0,0])
    cam_pos = np.reshape(cam_pos,(1,3))
    test = os.listdir(BasePath)
    builtin = []
    linear = []
    for image in sorted(test):
       images.append(image)
       
   
    for img,_ in enumerate(images[:-2]):
        # print(img)
        img1 = cv2.imread("%s/%s"%(BasePath,images[img]),0)
        img2 = cv2.imread("%s/%s"%(BasePath,images[img+1]),0)
        und1 = undistortImageToGray(img1,LUT)
        und2 = undistortImageToGray(img2,LUT)

        pts1, pts2 = features(und1,und2)
        # print(pts1.shape)
        if pts1.shape[0] <5:
            continue

        # F,_ = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
        E,_ = cv2.findEssentialMat(pts1, pts2, focal =K[0][0], pp=(K[0, 2],K[1, 2]), method=cv2.RANSAC, prob=0.999, threshold=0.5)
        
        
        _, R_new, C_new,_= cv2.recoverPose(E, pts1, pts2, focal=K[0, 0], pp=(K[0, 2], K[1, 2]))
       
        if np.linalg.det(R_new)<0:
            R_new = -R_new

        H2 = np.hstack((R_new,np.matmul(-R_new,C_new)))
        H2 = np.vstack((H2, [0, 0, 0, 1]))

        H1 = np.matmul(H1,H2)
       
        xpt = H1[0,3]
        zpt = H1[2,3]
 
        ax.scatter(-xpt, zpt, c = 'r')
        plt.draw()
        plt.pause(0.01)

        cv2.imshow('Test', img1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()

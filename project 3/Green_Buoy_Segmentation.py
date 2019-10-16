#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import math


# In[3]:


#Parameters got from GMM fitting. 
#Since we're testing for green but have to remove background noise which is a combination of RGB, we're testing for ll channels
def is_in_red(x):
    m1_red = 106.73365 
    m2_red = 172.67843
    var1_red = 266.25486
    var2_red = 943.23489

    sd1_red = np.sqrt(var1_red)
    sd2_red = np.sqrt(var2_red)
    
    l1_red = m1_red - 1*sd1_red
    h1_red = m1_red + 1*sd1_red 
    l2_red = m2_red - 1*sd2_red
    h2_red = m2_red + 1*sd2_red
    
    in_red = (x > l1_red and x < h1_red) or (x > l2_red and x < h2_red)
    return in_red

def is_in_green(x):
    m1_green = 245.8331
    m2_green = 239.31217
    var1_green = 476.897708
    var2_green = 41.68101
   

    sd1_green = np.sqrt(var1_green)
    sd2_green = np.sqrt(var2_green)
    
    l1_green = m1_green - 0.5*sd1_green
    h1_green = m1_green + 0.5*sd1_green 
    l2_green = m2_green - 1*sd2_green
    h2_green = m2_green + 1*sd2_green
    
    in_green = (x > l1_green and x < h1_green) or (x > l2_green and x < h2_green)
    return in_green

def is_in_blue(x):
    m1_blue = 143.43493
    m2_blue = 165.84652
    var1_blue = 72.30995
    var2_blue = 465.6737
   
    sd1_blue = np.sqrt(var1_blue)
    sd2_blue = np.sqrt(var2_blue)
    
    l1_blue = m1_blue - 1*sd1_blue
    h1_blue = m1_blue + 1*sd1_blue
    l2_blue = m2_blue - 1*sd2_blue
    h2_blue = m2_blue + 1*sd2_blue
    
    in_blue = (x > l1_blue and x < h1_blue) or (x > l2_blue and x < h2_blue)
    return in_blue


# ### Make sure the images and video are in the same directory as this code. 

# In[7]:


#Testing for particular images
test = cv2.imread("frame2.jpg")

red_channel = test[:,:,2]
green_channel = test[:,:,1]
blue_channel = test[:,:,0]   
for i in range(0,test.shape[0]):
    for j in range(0,test.shape[1]): 
        if is_in_red(red_channel[i,j]) and is_in_green(green_channel[i,j]) and is_in_blue(blue_channel[i,j]):
            test[i,j] = (255,255,255)
        else:
            test[i,j] = (0,0,0)
  
cv2.imshow('Output',test)
cv2.imwrite('Green_buoy_frame2.jpg',test)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[55]:


#Testing for the whole video
vid = cv2.VideoCapture("detectbuoy.avi")
while(True):
    ret, frame = vid.read()
    if not ret:
        break
    test=frame
    red_channel = test[:,:,2]
    green_channel = test[:,:,1]
    blue_channel = test[:,:,0]   
    for i in range(0,test.shape[0]):
        for j in range(0,test.shape[1]): 
            if is_in_green(green_channel[i,j]) and is_in_blue(blue_channel[i,j]) and is_in_red(red_channel[i,j]):
                test[i,j] = (255,255,255)
            else:
                test[i,j] = (0,0,0)
    kernel1 = np.ones((3,3), np.uint8) 
    img_erosion = cv2.erode(test, kernel1, iterations=1) 

    img_dilation = cv2.dilate(test, kernel1, iterations=1) 

    cv2.imshow('Output',img_dilation)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
vid.release()
cv2.destroyAllWindows()


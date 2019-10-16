#!/usr/bin/env python
# coding: utf-8

# ## Yellow buoy segmentation
# 

# In[26]:


import cv2
import numpy as np
import math


# In[27]:


#Parameters got from GMM fitting. Since we're testing for yellow, we have not considered the blue channel
def is_in_red(x):
    m_red = 225.6795823
    var_red = 409.225588
    
    sd_red = np.sqrt(var_red)
    l_red = m_red - 2*sd_red
    h_red = m_red + 2*sd_red
    in_red = x > l_red and x < h_red
    return in_red

def is_in_green(x):
    m_green = 254.46760403
    var_green = 261.3908766091778
    
    sd_green = np.sqrt(var_green)
    l_green = m_green - 1.5*sd_green
    h_green = m_green + 1.5*sd_green
    in_green = x > l_green and x < h_green
    return in_green


# ### Make sure the images and video are in the same directory as this code. 

# In[28]:


#Testing for particular images
test = cv2.imread("frame37.jpg")

red_channel = test[:,:,2]
green_channel = test[:,:,1]
blue_channel = test[:,:,0]   
for i in range(0,test.shape[0]):
    for j in range(0,test.shape[1]): 
        if is_in_red(red_channel[i,j]) and is_in_green(green_channel[i,j]):
            test[i,j] = (255,255,255)
        else:
            test[i,j] = (0,0,0)
kernel1 = np.ones((5,5), np.uint8)
img_erosion = cv2.erode(test, kernel1, iterations=1) 
img_dilation = cv2.dilate(img_erosion, kernel1, iterations=1) 
  
cv2.imshow('test',img_dilation)
cv2.imwrite('Yellow_buoy_frame37.jpg',test)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[30]:


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

            if is_in_red(red_channel[i,j]) and is_in_green(green_channel[i,j]):
                test[i,j] = (255,255,255)
            else:
                test[i,j] = (0,0,0)
  
    cv2.imshow('test',test)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
vid.release()
cv2.destroyAllWindows()


# In[ ]:





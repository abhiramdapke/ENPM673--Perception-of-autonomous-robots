




#!/usr/bin/env python
# coding: utf-8

# ## Orange model parameter



import cv2
import numpy as np
import imutils
import os







def is_in_red(x):
    m_red = 242.34579012445#238.27674182182795    
    var_red = 439.2317822289963
    
    sd_red = np.sqrt(var_red)
    l_red = m_red - 1*sd_red
    h_red = m_red + 1*sd_red
    in_red = x > l_red and x < h_red
    return in_red

def is_in_green(x):
    m1_green = 162.1367652285448
    m2_green = 199.7755662902594
    var1_green = 255.25873316800858
    var2_green = 478.6221902714627
    
    
    sd1_green = np.sqrt(var1_green)
    sd2_green = np.sqrt(var2_green)
    l1_green = m1_green - 2*sd1_green
    h1_green = m1_green + 2*sd1_green
    l2_green = m2_green - 1*sd2_green
    h2_green = m2_green + 1*sd2_green
    in_green = (x > l1_green and x < h1_green) or (x > l2_green and x < h2_green)
    return in_green

def is_in_blue(x):
    m1_blue = 96.80549600361515
    m2_blue = 126.23016658698457
    var1_blue = 99.64722456206842
    var2_blue = 344.3232121241507
    
    
    sd1_blue = np.sqrt(var1_blue)
    sd2_blue = np.sqrt(var2_blue)
    l1_blue = m1_blue - 1*sd1_blue
    h1_blue = m1_blue + 1*sd1_blue
    l2_blue = m2_blue - 0.5*sd2_blue
    h2_blue = m2_blue + 0.5*sd2_blue
    in_blue = (x > l1_blue and x < h1_blue) or (x > l2_blue and x < h2_blue)
    return in_blue

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

cwd = os.getcwd()
valid_images = [".jpg",".png",".jpeg"]
vid = cv2.VideoCapture("detectbuoy.avi")
vid=cv2.VideoCapture("input.avi")
while(True):
    ret, frame1=vid.read()
    ret, frame = vid.read()
    if not ret:
        break
    test=frame
    red_channel = test[:,:,2]
    green_channel = test[:,:,1]
    blue_channel = test[:,:,0]   
    for i in range(0,test.shape[0]):
        for j in range(0,test.shape[1]): 
            if is_in_red(red_channel[i,j]) and is_in_green(green_channel[i,j]) and is_in_blue(blue_channel[i,j]):
                test[i,j] = (255,255,255)
            else:
                test[i,j] = (0,0,0)
    test=cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(test, 60, 255, cv2.THRESH_BINARY)[1]
    cv2.convertScaleAbs(test)
    
    #ret, thresh = cv2.threshold(frame, 200, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #test=cv2.drawContours(test, contours, -1, (0,255,0), 3)
    test= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #test=cv2.Canny(test,100,200)
    for contour in contours:
        area=cv2.contourArea(contour)
        if area> 100 and area<300:
            frame=cv2.drawContours(frame1, contours, -1, (0,165,255), 3)
    test=imutils.grab_contours(test)
    cv2.imshow('test',frame1)
    out.write(frame1)
   # path2 = 'result_orange/result(' + str(f) + ').png'
    #cv2.imwrite('Result',test)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
vid.release()
out.release()
cv2.destroyAllWindows()






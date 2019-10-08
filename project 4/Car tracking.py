#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline
import glob


# In[2]:


def LucasKanadeAffine(currentFrame, nextFrame, rect, p0 = np.zeros(2)):

    
    threshold = 0.1
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
    Iy, Ix = np.gradient(nextFrame)
    dp = 1
    while np.linalg.norm(dp) > threshold:
        
        
        #warp image
        px, py = p0[0], p0[1]
        x1_w, y1_w, x2_w, y2_w = x1+px, y1-py, x2+px, y2-py
        
        x = np.arange(0, currentFrame.shape[0], 1)
        y = np.arange(0, currentFrame.shape[1], 1)
        
        c = np.linspace(x1, x2, abs(rect[2]-rect[0]))
        r = np.linspace(y1, y2, abs(rect[3]-rect[1]))
        cc, rr = np.meshgrid(c, r)
    
        cw = np.linspace(x1_w, x2_w, rect[2]-rect[0])
        rw = np.linspace(y1_w, y2_w, rect[3]-rect[1])
        ccw, rrw = np.meshgrid(cw, rw)
        
        spline = RectBivariateSpline(x, y, currentFrame)
        T = spline.ev(rr, cc)
        
        spline1 = RectBivariateSpline(x, y, nextFrame)
        warpImg = spline1.ev(rrw, ccw)
        
        #compute error image
        err = T - warpImg
        errImg = err.reshape(-1,1) 
        
        #compute gradient
        spline_gx = RectBivariateSpline(x, y, Ix)
        Ix_w = spline_gx.ev(rrw, ccw)

        spline_gy = RectBivariateSpline(x, y, Iy)
        Iy_w = spline_gy.ev(rrw, ccw)
        #I is (n,2)
        I = np.vstack((Ix_w.ravel(),Iy_w.ravel())).T
        
        #evaluate jacobian (2,2)
        jac = np.array([[1,0],[0,1]])
        
        #computer Hessian
        delta = I @ jac 
        #H is (2,2)
        H = delta.T @ delta
        
        #compute dp
        #dp is (2,2)@(2,n)@(n,1) = (2,1)
        dp = np.linalg.pinv(H) @ (delta.T) @ errImg
        
        #update parameters
        p0[0] += dp[0,0]
        p0[1] += dp[1,0]
        
    p = p0
    return p


# In[8]:


#Writes the template image for car
img = cv2.imread('frame0020.jpg')
rect=np.array([125,106,331,280])
img =cv2.rectangle(img,(rect[0], rect[1]),(rect[2], rect[3]),(0,0,255),3)
cv2.imwrite('Template_Car.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[6]:


def car_tracking():
    rect=np.array([125,103,336,280])
    filenames = glob.glob("*.jpg")
    filenames.sort()
    frames = [cv2.imread(img) for img in filenames]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in frames]
    for i in range(len(images)-1):
        currentFrame = images[i]
        nextFrame = images[i+1]
        frame = frames[i]
        p = LucasKanadeAffine(currentFrame,nextFrame,rect)
        img =cv2.rectangle(frame,(rect[0], rect[1]),(rect[2], rect[3]),(0,0,255),3)
        rect[0] += p[0]
        rect[1] -= p[1]
        rect[2] += p[0]
        rect[3] -= p[1]
        #cv2.imwrite('Out/' + str(i)+ '.jpg',img)   #Writes images into a folder named Out. Folder has to be created 
        cv2.imshow('Car_Tracking_Output', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #print(trans_point[0])
    cv2.destroyAllWindows()
    


# In[7]:


car_tracking()


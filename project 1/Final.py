
# coding: utf-8

# In[17]:


# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:53:49 2019

@author: sanke
"""



import cv2
import numpy as np


def Tag_no(test1):
    bin = []
    tag_no = 0
    col=np.array([test1[94,106], test1[106,106], test1[106,94], test1[94,94] ])
                  #Bottom Left   Bottom Right    Top Right       Top Left  
    for i in range(4):
        if np.min(col[i]>=240):
            bin.append(1)
        if np.max(col[i]<=10):
            bin.append(0)
#     if len(bin1)==4:
    for i in range(len(bin)):
            tag_no += (2**i)*bin[i]
    return tag_no


def order_points(pts,h,w):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    dst_temp = np.array([[0, 0],[w - 1, 0],[w - 1, h - 1],[0, h - 1]], dtype = "float32")
 
    # return the ordered coordinates
    return rect,dst_temp


def open_video(title):
    vid= cv2.VideoCapture(title)
    ref=cv2.imread('ref_marker.png',0)
    check =0
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #corners=cv2.goodFeaturesToTrack(gray,12,0.51,20)
        blurred=cv2.GaussianBlur(gray,(5,5), 0)
        ret, thresh = cv2.threshold(blurred, 200, 255, 0)
        f_con, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
        for contour in contours:
            area = cv2.contourArea(contour)
            if area<15000 and area >1000:
                img = cv2.drawContours(frame, contour, -1, (0,255,0),3)
        x=np.asarray(contours[1])
        leftmost = tuple(x[x[:,:,0].argmin()][0])
        rightmost = tuple(x[x[:,:,0].argmax()][0])
        topmost = tuple(x[x[:,:,1].argmin()][0])
        bottommost = tuple(x[x[:,:,1].argmax()][0])
        #img = cv2.drawContours(frame, contours, 0, (0,255,0),3)
        cv2.circle(img, leftmost, 8, (0, 0, 255), -1)
        cv2.circle(img, rightmost, 8, (0, 255, 0), -1)
        cv2.circle(img, topmost, 8, (255, 0, 0), -1)
        cv2.circle(img, bottommost, 8, (255, 255, 0), -1)
        #img = cv2.drawContours(frame, contours, 4, (0,255,0),3)
        #img = cv2.drawContours(frame, contours, 7, (0,255,0),3)
        h, w = ref.shape
        video=[leftmost,rightmost,topmost,bottommost]
        video = np.asarray(video)
        #given=[(0,0),(0,w),(w,h),(h,0)]
        given=[(w,h),(0,w),(h,0),(0,0)]
        given = np.asarray(given)
        rec,dst_temp =order_points(video,h,w)
        H = cv2.getPerspectiveTransform(rec, dst_temp)
        warped = cv2.warpPerspective(img, H, (w, h))
# warped[132,132]
        if(np.min(warped[57,57])>225):
            Rot=cv2.getRotationMatrix2D((h/2,w/2),180,1)
            temp=cv2.warpAffine(warped, Rot,(h,w))
            check=Tag_no(temp)
            
        elif(np.min(warped[57,132])>225):
            Rot=cv2.getRotationMatrix2D((h/2,w/2),90,1)
            temp=cv2.warpAffine(warped, Rot,(h,w))
            check=Tag_no(temp)
            
        elif(np.min(warped[132,132])>225):
            #Rot=cv2.getRotationMatrix2D((h/2,w/2),180,1)
            temp=cv2.warpAffine(warped, Rot,(h,w))
            check=Tag_no(temp)
            
        elif(np.min(warped[132,57])>225):
            Rot=cv2.getRotationMatrix2D((h/2,w/2),270,1)
            temp=cv2.warpAffine(warped, Rot,(h,w))
            check=Tag_no(temp)
            M, mask = cv2.findHomography(given, video, cv2.RANSAC, 2.0)
            lena = cv2.imread('Lena.png',0)
#         rows,cols = lena.shape
#         lena_pts=np.float32([[0,0],[rows,0],[rows,cols],[0,cols]])
#         trans = cv2.warpPerspective(lena, M, blurred.shape)
#         cv2.imshow("Len",trans)
#         cv2.waitKey(0)
#         canny=cv2.Canny(blurred, 100, 200)
        #if corners is not None:
          #  corners=np.int0(corners)
           # for corner in corners:
              #  x, y = corner.ravel()
              #  cv2.circle(frame, (x,y), 5, (0, 0, 255),-1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        detect=cv2.putText(img, "Tag no = "+str(check),(30,30), font, 0.5, (255,0,0),2, cv2.LINE_AA )
        cv2.imshow("Frame",detect)
        cv2.imshow("Frame_2",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
        #if not ret:
            break
    vid.release()
    cv2.destroyAllWindows()

open_video('Tag0.mp4')




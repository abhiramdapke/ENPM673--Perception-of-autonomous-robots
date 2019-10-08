# -*- coding: utf-8 -*-

import numpy as np 
import cv2


def warp(tmp,Ipo,p,Imgradx,Imgrady):
    wI = np.zeros(tmp.shape)
    wIgradx = np.zeros(tmp.shape)
    wIgrady = np.zeros(tmp.shape)
    p = np.reshape(p,(6,))
    pw = np.array([[1+p[0], p[2], p[4]],[p[1], 1+p[3], p[5]]])
    pw = np.reshape(pw,(2,3)) 
    for i in range(tmp.shape[1]):            
        for j in range(tmp.shape[0]):       
            x = np.array([i,j,1]).T
            x = np.reshape(x,(3,1))
            W = np.int16(np.round(np.matmul(pw,x)))
            a = Ipo[W[1],W[0]]
            igx = Imgradx[W[1],W[0]]
            igy = Imgrady[W[1],W[0]]
            wI[j,i] = a
            # print(wI[j,i])
            wIgradx[j,i] = igx
            wIgrady[j,i] = igy
    #print(wI.shape)
    return wI, wIgradx, wIgrady


def LucasKanadeAffine(tmp,Ipo,pprev,thresh,Imgradx,Imgrady):
    p = pprev
    delpnorm = thresh + 10
    while delpnorm > thresh:
        wI, wIgradx, wIgrady = warp(tmp,Ipo,p,Imgradx,Imgrady)
        s1 = np.zeros([6,1])
        s2 = np.zeros([6,6])
        for i in range(tmp.shape[1]):            
            for j in range(tmp.shape[0]):       
                wJ = np.array([[i,0,j,0,1,0],[0,i,0,j,0,1]])
                ig = np.array([wIgradx[j,i],wIgrady[j,i]])
                b1 = np.matmul(ig,wJ)
                b2 = tmp[j,i] - wI[j,i]
                # b2 = np.absolute(t[j,i] - wI[j,i])
                b1 = np.reshape(b1,(1,6))
                b2 = np.reshape(b2,(1,1))
                s1 = s1 + b1.T*b2
                s2 = s2 + b1.T*b1
        H = s2
        sdpu = s1
        delp = np.matmul(np.linalg.pinv(H),sdpu)
        p = p + delp
        delpnorm = np.linalg.norm(delp)
        #print('dpnorm',delpnorm)
    return p

img1 = cv2.imread('0020.jpg')
xleft=124
xbottom=173
yleft=93
ybottom=150
temp = img1[yleft:ybottom,xleft:xbottom]
temp = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
cv2.imwrite('vase_template.jpg',temp)
p = np.zeros([6,1])
p[4] = xleft
p[5] = yleft
pw = np.array([[1+p[0], p[2], p[4]],[p[1], 1+p[3], p[5]]])
pw = np.reshape(pw,(2,3))
c1 = np.matmul(pw,np.reshape([0,0,1],(3,1)))
c2 = np.matmul(pw,np.reshape([xbottom-xleft,0,1],(3,1)))
c3 = np.matmul(pw,np.reshape([xbottom-xleft,ybottom-yleft,1],(3,1)))
c4 = np.matmul(pw,np.reshape([0,ybottom-yleft,1],(3,1)))
pprev = p
frame_no = 21
while True:
    if frame_no < 100:
        frame = cv2.imread('00'+str(frame_no)+'.jpg')
    else:
        frame = cv2.imread('0'+str(frame_no)+'.jpg')
    I = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    Igradx = cv2.Sobel(I,cv2.CV_64F,1,0,ksize =3)
    Igrady = cv2.Sobel(I,cv2.CV_64F,0,1,ksize =3)
    thresh = 0.005
    pnew = LucasKanadeAffine(temp,I,pprev,thresh,Igradx,Igrady)
    #print('pnew',pnew)
    pw = np.array([[1+pnew[0], pnew[2], pnew[4]],[pnew[1], 1+pnew[3], pnew[5]]])
    pw = np.reshape(pw,(2,3))
    c1 = np.matmul(pw,np.reshape([0,0,1],(3,1)))
    c2 = np.matmul(pw,np.reshape([xbottom-xleft,0,1],(3,1)))
    c3 = np.matmul(pw,np.reshape([xbottom-xleft,ybottom-yleft,1],(3,1)))
    c4 = np.matmul(pw,np.reshape([0,ybottom-yleft,1],(3,1)))
    corners = np.int64(np.hstack([c1,c2,c3,c4]))
    #print(corners)
    x1 = min(corners[0,:])
    y1 = min(corners[1,:])
    x2 = max(corners[0,:])
    y2 = max(corners[1,:])
    #print(x1,y1)

    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
    cv2.imshow('I',frame)
    #cv2.imwrite('Vase/' + str(frame_no) + '.jpg',frame)     #Uncomment this if ypu want to write the output imags in a folder named Vase. 
    if frame_no == 169:
        break
    pprev = pnew
    frame_no = frame_no + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
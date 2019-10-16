
# coding: utf-8

import cv2
import numpy as np


# In[2]:


#Camera Matrix
K =  np.array([[  1.15422732e+03,   0.00000000e+00,   6.71627794e+02],
      [  0.00000000e+00,   1.14818221e+03,   3.86046312e+02],
      [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

#Distortion Coefficients
dist = np.array([[ -2.42565104e-01,  -4.77893070e-02,  -1.31388084e-03,  -8.79107779e-05, 2.20573263e-02]])


def undistorted(image):
    dst = cv2.undistort(image, K, dist, None, K)
    return dst

def sobel(l_channel):
# Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    return scaled_sobel


def frame_process(img, s_thresh=(95, 255), sx_thresh=(15, 255)):
    img = undistorted(img)
    img = np.copy(img)
    # Converts to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    
    s_channel = hls[:,:,2]
   
    scaled_sobel= sobel(l_channel)
    # Threshold x gradient
    sobelxbinary = np.zeros_like(scaled_sobel)
    sobelxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    sobel_binary = np.zeros_like(s_channel)
    sobel_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    combined_binary = np.zeros_like(sobelxbinary)
    combined_binary[(sobel_binary == 1) | (sobelxbinary == 1)] = 1
    return combined_binary

def perspective_warp(image):
    image_size = np.float32([(image.shape[1],image.shape[0])])
    src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
    dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])
    dst_size=(image.shape[1], image.shape[0])
    src = src* image_size
    dst = dst * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(image, M, dst_size)
    return warped

def inv_perspective_warp(img):
    dst_size=(img.shape[1], img.shape[0])
    src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
    dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])
    image_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* image_size
    dst = dst * np.float32(dst_size)
    # Calculates the perspective transform matrix
    persp_trans = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, persp_trans, dst_size)
    return warped

def get_histogram(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist







left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]

def sliding_window(img, nwindows=12, margin=150, minpix = 1, draw_windows=True):
   
    left_fit_= np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    histogram = get_histogram(img)
    
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    
    
    window_height = np.int(img.shape[0]/nwindows)
   
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    
    
    left_lane_inds = []
    right_lane_inds = []

    
    for window in range(nwindows):
        
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        if draw_windows == True:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (100,255,255), 3) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (100,255,255), 3) 
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))


    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    l_fit = np.polyfit(lefty, leftx, 2)
    r_fit = np.polyfit(righty, rightx, 2)
    
    right_a.append(r_fit[0])
    right_b.append(r_fit[1])
    right_c.append(r_fit[2])
    
    left_a.append(l_fit[0])
    left_b.append(l_fit[1])
    left_c.append(l_fit[2])
    
    
    
    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])
    
    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])
    
    # Generate x and y values for plotting
    disp_y = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit_[0]*disp_y**2 + left_fit_[1]*disp_y + left_fit_[2]
    right_fitx = right_fit_[0]*disp_y**2 + right_fit_[1]*disp_y + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
    
    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), disp_y

def curvature(img, leftx, rightx):
    disp_y = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(disp_y)
    # meters per pixel in y dimension
    ym_per_pix = 30.5/720 
    # meters per pixel in x dimension
    xm_per_pix = 3.7/720 

    
    l_f_cr = np.polyfit(disp_y*ym_per_pix, leftx*xm_per_pix, 2)
    r_f_cr = np.polyfit(disp_y*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculates the new radii of curvature
    l_c_rad = ((1 + (2*l_f_cr[0]*y_eval*ym_per_pix + l_f_cr[1])**2)**1.5) / np.absolute(2*l_f_cr[0])
    r_c_rad = ((1 + (2*r_f_cr[0]*y_eval*ym_per_pix + r_f_cr[1])**2)**1.5) / np.absolute(2*r_f_cr[0])

    return (l_c_rad, r_c_rad)

def draw_lanes(img, left_fit, right_fit):
    disp_y = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_img = np.zeros_like(img)
    
    l = np.array([(np.vstack([left_fit, disp_y]).T)])
    r = np.array([np.flipud(np.vstack([right_fit, disp_y]).T)])
    points = np.hstack((l, r))
    
    cv2.fillPoly(color_img, np.int_(points), (0,0,255))
    inv_perspective = inv_perspective_warp(color_img)
    inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.6, 0)
    return inv_perspective


vid= cv2.VideoCapture("project_video.mp4")
while True:
    ret, frame = vid.read()
    if not ret:
        break
    
    img=cv2.GaussianBlur(frame,(5,5), 0)
    
    dst = undistorted(img)
    dst = frame_process(img)
    dst = perspective_warp(dst)
    
    out_img, curves, lanes, disp_y = sliding_window(dst)
    
    
    curverad=curvature(img, curves[0],curves[1])
    diff=curverad[0]-curverad[1]

    display = draw_lanes(frame, curves[0], curves[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display,'Left curvature:'+str(curverad[0]),(10,50), font, 1,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(display,'Right curvature:'+str(curverad[1]),(10,100), font, 1,(255,0,0),2,cv2.LINE_AA)
    if (diff>100 and diff<1000 and curverad[0<3000 and curverad[1]<3000]):
            cv2.putText(display,'Left Turn',(10,150), font, 1,(0,255,0),2,cv2.LINE_AA)
    if (diff>-1000 and diff <-100 and curverad[0]<3000 and curverad[1]<3000):
            cv2.putText(display,'Right Turn',(10,150), font, 1,(0,255,0),2,cv2.LINE_AA)
    cv2.imshow("Lanes", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()


import cv2
import numpy as np
# import matplotlib.pyplot as plt


def FindCorrespondence(img1, img2):
#   Initiate SIFT detector
	orb = cv2.ORB_create(200)

	# find the keypoints and descriptors with SIFT
	kp1, des1 = orb.detectAndCompute(img1, None)
	kp2, des2 = orb.detectAndCompute(img2, None)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
	matches = bf.match(des1, des2)

	matches = sorted(matches, key = lambda x: x.distance)

	#img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches,None, flags = 2)
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
	    (x1,y1) = kp1[img1_idx].pt
	    (x2,y2) = kp2[img2_idx].pt

	    # Append to each list
	    list_kp1.append((x1, y1))
	    list_kp2.append((x2, y2))


	list_kp1 = np.array(list_kp1)
	list_kp2 = np.array(list_kp2)
	return list_kp1, list_kp2


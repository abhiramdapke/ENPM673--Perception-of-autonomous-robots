import numpy as np
import random
import cv2
from EstimateFundamentalMatrix import*
import EstimateFundamentalMatrix as EFM




def RANSACforF(pts1, pts2, corr_list, thresh=0.9):
    num_points_F = 8
    maxInliers = []
    finalF = None
    for i in range(1000):
        corr = []
        # find n random points to calculate a homography
        for n in range(num_points_F):
            corr.append(corr_list[random.randrange(0, len(corr_list))])

        # Calculate Fundamental Matrix function on those points
        f = EFM.EstimateFundamentalMatrix(pts1, pts2, num_points_F)
        inliers = []

        for i in range(len(corr_list)):
            d = geometricDistance(corr_list[i], f)
            if d < 0.2:
                inliers.append(corr_list[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalF = f
        print("Corr size: ", len(corr_list), " NumInliers: ",
        len(inliers), "Max inliers: ", len(maxInliers))

        if len(maxInliers) > (len(corr) * thresh):
            break
    FCheck, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    return FCheck, maxInliers

def geometricDistance(correspondence, f):
	p1 = np.array([correspondence[0], correspondence[1], 1])
	p2 = np.array([correspondence[2], correspondence[3], 1])
	estimate = p2@f@p1.T



	return np.linalg.norm(estimate)

# F,mI = RANSACforF(corr_list)
# print(F)


# FCheck, mask = cv2.findFundamentalMat(points_img1, points_img2, cv2.FM_RANSAC)
# print("F from openCV: \n", FCheck)


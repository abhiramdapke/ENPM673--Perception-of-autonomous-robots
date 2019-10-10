import glob
import numpy as np
import cv2
import tkinter
import matplotlib.pyplot as plt
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
from FindCorrespondence import*
from EstimateFundamentalMatrix import*
from RANSACforF import*
from EstimateEssentialMatrix import*
from EstimateCameraPose import*
from LinearTriangulation import*
from Disambiguation import*
from mpl_toolkits.mplot3d import Axes3D

fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('./model')
cameraMatrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

filenames = glob.glob("stereo/centre/*.png")
filenames.sort()
j = 1
t_f = np.array([[0],[0],[0]])
t1 = t_f
H1 = np.array(np.eye(4))

scale = 1
print(t_f[1,0])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(filenames) - 1):
    if i<=40:
        continue

    f = filenames[i]
    f1 = filenames[i + 1]
    img1 = cv2.imread(f, 0)
    img2 = cv2.imread(f1, 0)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BAYER_GR2RGB)
    img1 = UndistortImage(img1, LUT)    
    img2 = cv2.cvtColor(img2, cv2.COLOR_BAYER_GR2RGB)
    img2 = UndistortImage(img2, LUT)


    img1GrayScale = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2GrayScale = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    pts1, pts2 = FindCorrespondence(img1GrayScale, img2GrayScale)
    corr_list = np.concatenate((pts1, pts2), 1)

    F, mI = RANSACforF(pts1, pts2, corr_list)

    E = EstimateEssentialMatrix(cameraMatrix, F)


    P1, P2, P3, P4, (C1, R1), (C2, R2), (C3, R3), (C4, R4) = EstimateCameraPose(E, cameraMatrix)


    C_new, R_new = Disambiguate(pts1, pts2, cameraMatrix, P1, P2, P3, P4, R1, R2, R3, R4, C1, C2, C3, C4)

    if np.linalg.det(R_new)<0:
        R_new = -R_new

    H2 = np.hstack((R_new, np.matmul(-R_new, C_new)))
    H2 = np.vstack((H2, [0, 0, 0, 1]))

    H1 = np.matmul(H1, H2)

    xpt = H1[0, 3]
    zpt = H1[2, 3]
 
    ax.scatter(-xpt, zpt, color='r', marker='o')
    plt.pause(0.05)
    cv2.imshow("dfr", img1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cv2.destroyAllWindows()

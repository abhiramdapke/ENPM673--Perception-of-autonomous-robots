from LinearTriangulation import*
import numpy as np 

def Disambiguate(points_img1, points_img2, K, P1, P2, P3, P4, R1, R2, R3, R4, C1, C2, C3, C4):
    X1 = LinearTriangulation(np.array([[0],[0],[0]]), np.eye(3), P1, points_img1, points_img2, K)
    X2 = LinearTriangulation(np.array([[0],[0],[0]]), np.eye(3), P2, points_img1, points_img2, K)
    X3 = LinearTriangulation(np.array([[0],[0],[0]]), np.eye(3), P3, points_img1, points_img2, K)
    X4 = LinearTriangulation(np.array([[0],[0],[0]]), np.eye(3), P4, points_img1, points_img2, K)

    R3_1 = R1[2, :]
    R3_2 = R2[2, :]
    R3_3 = R3[2, :]
    R3_4 = R4[2, :]

    R3_1 = np.array(R3_1).reshape((len(R3_1), -1))
    R3_2 = np.array(R3_2).reshape((len(R3_2), -1))
    R3_3 = np.array(R3_3).reshape((len(R3_3), -1))
    R3_4 = np.array(R3_4).reshape((len(R3_4), -1)) 

    test1 = (X1-C1.T)@R3_1
    test2 = (X2-C2.T)@R3_2
    test3 = (X3-C3.T)@R3_3
    test4 = (X4-C4.T)@R3_4


    sum1 = sum(x>0 for x in test1)
    sum2 = sum(x>0 for x in test2)
    sum3 = sum(x>0 for x in test3)
    sum4 = sum(x>0 for x in test4)
    maxSum = np.max([sum1, sum2, sum3, sum4])

    if maxSum == sum1:
        return C1, R1
    if maxSum == sum2:
        return C2, R2
    if maxSum == sum3:
        return C3, R3
    if maxSum == sum4:
        return C4, R4

# print(Disambiguate())





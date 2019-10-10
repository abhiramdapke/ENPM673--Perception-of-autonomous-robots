import numpy as np


def EstimateFundamentalMatrix(points_img1, points_img2, n):
    x1 = np.array(points_img1)[:,0]
    #print(x1)
    #x1 = x1[0:n]
    x1 = np.array(x1).reshape((len(x1), -1))
    x2 = np.array(points_img2)[:,0]
    #x2 = x2[0:n]
    x2 = np.array(x2).reshape((len(x2), -1))
    y1 = np.array(points_img1)[:,1]
    #y1 = y1[0:n]
    y1 = np.array(y1).reshape((len(y1), -1))
    y2 = np.array(points_img2)[:,1]
    #y2 = y2[0:n]
    y2 = np.array(y2).reshape((len(y2), -1))

    C1 = x1 * x2
    C2 = x1 * y2
    C3 = x1
    C4 = y1 * x2
    C5 = y1 * y2
    C6 = y1
    C7 = x2
    C8 = y2
    C9 = np.ones((len(x1), 1))



    A = np.array([C1, C2, C3, C4, C5, C6, C7, C8, C9])
    A = np.concatenate((C1, C2, C3, C4, C5, C6, C7, C8, C9), 1)

    U, S, V_trans = np.linalg.svd(A)

    F = (V_trans[-1].reshape(3, 3)).T #or transpose, remember to check
    
    U1, S1, V_trans1 = np.linalg.svd(F)
    S1[-1] = 0
    #S1 = np.diag(S1)
    F = U1@np.diag(S1)@V_trans1

    # F1 = EstimateFundamentalMatrix(points_img1, points_img2, 15)


    return F

# F = EstimateFundamentalMatrix(points_img1, points_img2, 15)

# print(F)


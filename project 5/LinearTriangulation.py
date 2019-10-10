from EstimateCameraPose import*
import numpy as np

# P1, P2, P3, P4, (C1, R1), (C2, R2), (C3, R3), (C4, R4) = EstimateCameraPose()


def LinearTriangulation(C1, R1, P2, x1, x2, cameraMatrix):
    # p1 = P1[0, :]
    # p2 = P1[1, :]
    # p3 = P1[2, :]

    # pts1Norm = np.linalg.norm(pts1, axis=1)
    # pts1 = pts1 / pts1Norm
    # one = np.ones((len(pts), 1))

    I = np.eye(3)
    I1 = np.array(I[:, 0]).reshape(len(I[:, 0]), -1)
    I2 = np.array(I[:, 1]).reshape(len(I[:, 1]), -1)
    I3 = np.array(I[:, 2]).reshape(len(I[:, 2]), -1)

    P1 = cameraMatrix@R1@np.concatenate((I1, I2, I3, -C1), 1)

    N = len(x1)
    X = np.zeros((N, 4))

    for i in range(N):
        A = [x1[i, 1] * P1[2, :] - P1[1, :], P1[0, :] - x1[i, 0] * P1[2, :], x2[i, 1] * P2[2, :] - P2[1, :], P2[0, :] - x2[i, 0] * P2[2, :]]
        U, S, Vtrans = np.linalg.svd(A)
        X[i, :] = Vtrans[3, :] / Vtrans[3, 3]
    X = X[:, 0:3]
    return X




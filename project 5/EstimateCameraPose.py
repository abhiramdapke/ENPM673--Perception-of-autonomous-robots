# from EstimateEssentialMatrix import*
import numpy as np


def EstimateCameraPose(E, cameraMatrix):

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    U, S, Vtrans = np.linalg.svd(E)

    C1 = U[:, 2]
    C2 = -U[:, 2]
    C3 = U[:, 2]
    C4 = -U[:, 2]

    R1 = U@W@Vtrans
    if np.linalg.det(R1) == -1:
        C1 = -C1
        C2 = -C2
        R1 = -R1

    R3 = U@W.T@Vtrans
    if np.linalg.det(R3) == -1:
        C3 = -C3
        C4 = -C4
        R3 = -R3

    C1 = np.array(C1).reshape(len(C1), -1)
    C2 = np.array(C2).reshape(len(C2), -1)
    C3 = np.array(C3).reshape(len(C3), -1)
    C4 = np.array(C4).reshape(len(C4), -1)

    R2 = R1
    R4 = R3
    I = np.eye(3)
    I1 = np.array(I[:, 0]).reshape(len(I[:, 0]), -1)
    I2 = np.array(I[:, 1]).reshape(len(I[:, 1]), -1)
    I3 = np.array(I[:, 2]).reshape(len(I[:, 2]), -1)

    P1 = cameraMatrix@R1@np.concatenate((I1, I2, I3, -C1), 1)
    P2 = cameraMatrix@R2@np.concatenate((I1, I2, I3, -C2), 1)
    P3 = cameraMatrix@R3@np.concatenate((I1, I2, I3, -C3), 1)
    P4 = cameraMatrix@R4@np.concatenate((I1, I2, I3, -C4), 1)

    return P1, P2, P3, P4, (C1, R1), (C2, R2), (C3, R3), (C4, R4)

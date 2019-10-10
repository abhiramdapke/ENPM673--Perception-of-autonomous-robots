import numpy as np


def EstimateEssentialMatrix(cameraMatrix, F):

	E = cameraMatrix.T@F@cameraMatrix

	U, S, Vtrans = np.linalg.svd(E)
	Snew = np.diag([1,1,0])

	Enew = U@Snew@Vtrans
	return Enew


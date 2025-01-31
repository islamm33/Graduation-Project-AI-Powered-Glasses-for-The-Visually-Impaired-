import scipy
import cv2
# Load the .mat file
mat = scipy.io.loadmat('calibration_params.mat')

# Extract the parameters
cameraMatrix1 = mat['cameraMatrix1']
cameraMatrix2 = mat['cameraMatrix2']
distCoeffs1 = mat['distCoeffs1']
distCoeffs2 = mat['distCoeffs2']
R = mat['R']
T = mat['T']

# Create a FileStorage object
fs = cv2.FileStorage('MATLAB_calibration_params.xml', cv2.FILE_STORAGE_WRITE)

# Write the parameters to the file
fs.write('cameraMatrix1', cameraMatrix1)
fs.write('cameraMatrix2', cameraMatrix2)
fs.write('distCoeffs1', distCoeffs1)
fs.write('distCoeffs2', distCoeffs2)
fs.write('R', R)
fs.write('T', T)

# Release the file
fs.release()
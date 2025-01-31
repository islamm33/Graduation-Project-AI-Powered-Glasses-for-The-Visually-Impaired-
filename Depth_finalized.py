
import cv2
import numpy as np


# Load camera parameters from XML file
cv_file = cv2.FileStorage('MATLAB_calib2.xml', cv2.FileStorage_READ)

cameraMatrix1 = cv_file.getNode('cameraMatrix1').mat()
distCoeffs1 = cv_file.getNode('distCoeffs1').mat()
cameraMatrix2 = cv_file.getNode('cameraMatrix2').mat()
distCoeffs2 = cv_file.getNode('distCoeffs2').mat()
R = cv_file.getNode('R').mat()
T = cv_file.getNode('T').mat()

# print(cameraMatrix1.shape)
# print(distCoeffs1.shape)
# print(cameraMatrix2.shape)
# print(distCoeffs2.shape)
# print(R.shape)
# print(T.shape)

# Set desired width, height 
desired_width = 1280
desired_height = 480
imageSize = (640, 480)

output_width = 640
output_height = 480

focal_lenght = 267
baseline = 0.062
y_min = 1
y_max= 100
x_min = 2
x_max = 100


# Compute rectification transforms for each camera
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T)

# Generate the undistortion and rectification maps
stereoMapL_x, stereoMapL_y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, cv2.CV_32FC1)
stereoMapR_x, stereoMapR_y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, cv2.CV_32FC1)

# # Initialize the video capture object with desired settings
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use 0 for the default camera, or provide the camera index if you have multiple cameras
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# stereo = cv2.StereoBM_create(numDisparities=32, blockSize=21)

# while(cap.isOpened()):

#     # Read the stereo pair from the single camera
#     success, frame = cap.read()
#     if not success:
#         break

#     # Split the stereo pair into two frames
#     frame_left = frame[:, :frame.shape[1]//2]
#     frame_right = frame[:, frame.shape[1]//2:]

frame_right = cv2.imread(r'D:\LocalDisk\GP\Sprints\StereoVisionDepthEstimation\new_images\stereoRight\imageR0.png')
frame_left = cv2.imread(r'D:\LocalDisk\GP\Sprints\StereoVisionDepthEstimation\new_images\stereoLeft\imageL0.png')
    # Undistort and rectify images
frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
cv2.imshow("Frame Right", frame_right)
cv2.imshow("Frame left", frame_left)
# Center the remapped images
# center_x = frame_left.shape[1] // 2
# center_y = frame_left.shape[0] // 2
# half_width = frame_left.shape[1] // 4
# half_height = frame_left.shape[0] // 4

# frame_left_centered = frame_left[center_y - half_height:center_y + half_height,
#                                     center_x - half_width:center_x + half_width]

# frame_right_centered = frame_right[center_y - half_height:center_y + half_height,
#                                     center_x - half_width:center_x + half_width]

# # Resize the centered remapped images
# frame_left = cv2.resize(frame_left_centered, (output_width, output_height))
# frame_right = cv2.resize(frame_right_centered, (output_width, output_height))




frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
# # Initialize the StereoBM object
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
# Set SGBM parameters (adjust these values as needed)
stereo = cv2.StereoSGBM_create(minDisparity=0, 
                             numDisparities=64,
                             blockSize=5,
                             P1=8,
                             P2=32,
                             disp12MaxDiff=1,
                             uniquenessRatio=10,
                             speckleWindowSize=5,
                             speckleRange=32,
                             preFilterCap=63,
                             mode=cv2.StereoSGBM_MODE_SGBM  # Set mode to SGBM
                             )

# Compute the disparity map
disparity_map = stereo.compute(frame_left, frame_right)

# Normalize the disparity map for visualization
disparity_map_normalized = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Save the disparity map
cv2.imwrite("disparity_map_Not_Centred.png", disparity_map_normalized)

print("Disparity map saved as disparity_map_Not_Centred.png")

cv2.imshow("Dosparity Map", disparity_map_normalized)

roi = disparity_map_normalized[y_min:y_max, x_min:x_max]  # Region of interest (bounding box)

average_disparity = np.mean(roi)

depth = focal_lenght*baseline/average_disparity 
print (f'Depth of object is {depth} meters ')

#cv2.imshow("Disparityyy", disparity_map_normalized ) 

# Estimate new camera matrices for undistortion

# distCoeffs1_array = np.asarray(distCoeffs1, dtype=np.float32)
# distCoeffs2_array = np.asarray(distCoeffs2, dtype=np.float32)
# final_K_left = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(cameraMatrix1, distCoeffs1_array, frame_left.shape[:2][::-1], np.eye(3), balance=1.0)
# final_K_right = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(cameraMatrix2, distCoeffs2_array, frame_right.shape[:2][::-1], np.eye(3), balance=1.0)

#     # Compute undistortion and rectification maps
# map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(final_K_left, distCoeffs1_array, np.eye(3), final_K_left, frame_left.shape[:2][::-1], cv2.CV_32FC1)
# map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(final_K_right, distCoeffs2_array, np.eye(3), final_K_right, frame_right.shape[:2][::-1], cv2.CV_32FC1)

#     # Apply fisheye undistortion to left and right frames
# undistorted_left = cv2.remap(frame_left, map1_left, map2_left, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
# undistorted_right = cv2.remap(frame_right, map1_right, map2_right, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

#     # Display the undistorted frames
# cv2.imshow("Undistorted Left", undistorted_left)
# cv2.imshow("Undistorted Right", undistorted_right)

while cv2.waitKey(0) != 27:  # Wait until Escape key (27) is pressed
    pass

    # frame_right = cv2.fisheye.undistortImage(frame_right, cameraMatrix2, distCoeffs2)
    # frame_left = cv2.fisheye.undistortImage(frame_left, cameraMatrix1, distCoeffs1)

   # Center the remapped images
    # center_x = frame_left.shape[1] // 2
    # center_y = frame_left.shape[0] // 2
    # half_width = frame_left.shape[1] // 4
    # half_height = frame_left.shape[0] // 4

    # frame_left_centered = frame_left[center_y - half_height:center_y + half_height,
    #                                  center_x - half_width:center_x + half_width]

    # frame_right_centered = frame_right[center_y - half_height:center_y + half_height,
    #                                    center_x - half_width:center_x + half_width]

    # # Resize the centered remapped images
    # frame_left_resized = cv2.resize(frame_left_centered, (output_width, output_height))
    # frame_right_resized = cv2.resize(frame_right_centered, (output_width, output_height))

    # # Display the resized remapped images
    # cv2.imshow("Resized Remapped Left", frame_left_resized)
    # cv2.imshow("Resized Remapped Right", frame_right_resized)
                     
    # # Show the frames
    # cv2.imshow("frame right", frame_right) 
    # cv2.imshow("frame left", frame_left)

    #  # Convert to grayscale
    # left_gray = cv2.cvtColor(frame_left_resized, cv2.COLOR_BGR2GRAY)
    # right_gray = cv2.cvtColor(frame_right_resized, cv2.COLOR_BGR2GRAY)

    # # Compute the disparity map
    # disparity_map = stereo.compute(left_gray, right_gray)

    # # Display the disparity map
    # cv2.imshow('Disparity Map', disparity_map)

    # Hit "q" to close the window
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release and destroy all windows before termination
# cap.release()
# cv2.destroyAllWindows()


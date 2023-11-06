import os
import cv2
import numpy as np

# set chessboard row
width = 8
height = 11

def findCorner(folder_path):

    if not folder_path.strip():
        print("Q1-1 Warning : 請選擇路徑")
        return True

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    for image in image_files:

        # 灰階
        image = folder_path + "/" + image
        image = cv2.imread(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 設定行列
        corner_width = 11
        corner_height = 8

        # 查角落
        ret, corners = cv2.findChessboardCorners(gray, (corner_width, corner_height), None)

        if ret:
            # 優化找的像素點
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

            # 畫角落
            cv2.drawChessboardCorners(image, (corner_width, corner_height), corners, ret)

            # show image
            chess_height, chess_width, _ = image.shape
            cv2.namedWindow("Chessboard", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Chessboard", int(chess_width * 0.6), int(chess_height * 0.6))
            cv2.imshow('Chessboard', image)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        else:
            print("沒找到，可憐")


def findIntrMatrix(folder_path):
    if not folder_path.strip():
        print("Q1-2 Warning : 請選擇路徑")
        return True

    # set criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(7,10,0)
    objp = np.zeros((11 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3D point in real world space
    imgpoints = []  # 2D points in image plane

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    for image in image_files:
        image = folder_path + "/" + image
        image = cv2.imread(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(corners2)

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)

    print("Intrinsic Matrix:")
    print(mtx)


def findExtrMatrix(folder_path, num_of_Image):
    if not folder_path.strip():
        print("Q1-3 Warning : 請選擇路徑")
        return True
    
    if num_of_Image == 100:
        print("Q1-3 Warning : 請輸入第幾張圖片")
        return True

    # set criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(7,10,0)
    objp = np.zeros((11 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3D point in real world space
    imgpoints = []  # 2D points in image plane

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    for image in image_files:
        image = folder_path + "/" + image
        image = cv2.imread(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(corners2)

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)
    
    # Generate extrinsic matrices for all images
    extrinsic_matrices = []
    for rvec, tvec in zip(rvecs, tvecs):
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        extrinsic_matrix = np.hstack((rotation_matrix, tvec))
        extrinsic_matrices.append(extrinsic_matrix)

    print(f"第 {num_of_Image} 張圖的Extrinsic Matrices:\n{extrinsic_matrices[num_of_Image - 1]}")


def findDistMatrix(folder_path):

    if not folder_path.strip():
            print("Q1-4 Warning : 請選擇路徑")
            return True

    # set criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(7,10,0)
    objp = np.zeros((11 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3D point in real world space
    imgpoints = []  # 2D points in image plane

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    for image in image_files:
        image = folder_path + "/" + image
        image = cv2.imread(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(corners2)

    # # Get the image shape (assuming all images have the same size)
    # image_shape = gray.shape[::-1]

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)

    print("Distortion  Matrix:")
    distList = []
    for i in dist:
        distList.append(i)
    print(distList)

def showUndistor(folder_path):

    if not folder_path.strip():
        print("Q1-5 Warning : 請選擇路徑")
        return True

    # set criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(7,10,0)
    objp = np.zeros((11 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3D point in real world space
    imgpoints = []  # 2D points in image plane

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    for image in image_files:
        image = folder_path + "/" + image
        image = cv2.imread(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(corners2)

    # # Get the image shape (assuming all images have the same size)
    # image_shape = gray.shape[::-1]

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)

    # # create Mapping
    # map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, None, (width, height), cv2.DIST_L2)
    # create new_mtx
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
    # show image
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    for image in image_files:
        image = folder_path + "/" + image

        # use Mapping
        # undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        cv2_image = cv2.imread(image)
        undistorted_image = cv2.undistort(cv2_image, mtx, dist, None, new_mtx)

        merged_image = cv2.hconcat([cv2_image, undistorted_image])

        # adjust window
        cv2.namedWindow("Merged Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Merged Image", 1600, 800)

        cv2.imshow("Merged Image", merged_image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
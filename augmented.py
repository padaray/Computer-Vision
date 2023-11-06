import os
import cv2
import yaml
import re
import numpy as np

# set chessboard row
width = 8
height = 11

# get letter imformation in alphabet_lib_onboard.txt
def loadAlphabetLibrary(folder_path, letter):
    yamlpath = folder_path + '/Q2_lib/alphabet_lib_onboard.txt'
    cv_file = cv2.FileStorage(yamlpath, cv2.FILE_STORAGE_READ)
    return cv_file.getNode(letter).mat()

# get letter imformation in alphabet_lib_vertical.txt
def loadAlphabetVirtical(folder_path, letter):
    yamlpath = folder_path + '/Q2_lib/alphabet_lib_vertical.txt'
    cv_file = cv2.FileStorage(yamlpath, cv2.FILE_STORAGE_READ)
    return cv_file.getNode(letter).mat()



def showOnBoard(folder_path, input_text):

    if not folder_path.strip():
        print("Q2-1 Warning : 請選擇路徑")
        return True
    
    if input_text == 100:
        print("Q2-1 Warning : 請輸入文字")
        return True

    # set criteria
    chessboard_with_word = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(7,10,0)
    objp = np.zeros((11 * 8, 3), np.float32)
    for i in range(0, 8):
        for j in range(0, 11):
            objp[i * 11 + j] = [j, i, 0]

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3D point in real world space
    imgpoints = []  # 2D points in image plane

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    for image in image_files:
        image = folder_path + "/" + image
        image = cv2.imread(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (height, width), None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(corners2)

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)

    # create new_mtx
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
    anchor = [[7, 5, 0],[4, 5, 0],[1, 5, 0],[7, 2, 0],[4, 2, 0],[1, 2, 0]]
    
    # deal with every image
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    for i, image in enumerate(image_files):
        image = folder_path + "/" + image
        cv2_image = cv2.imread(image, cv2.COLOR_BGR2GRAY)
        
        # foreach input text
        for j, letter in enumerate(input_text):
            letter_shape = []
            latter_mat = loadAlphabetLibrary(folder_path, letter)
            
            # 每筆畫加上起始位子(讓每個字母在不同位子)
            for line in latter_mat:
                line += anchor[j]
                print("word_shape印出來長這樣BBBBBB : \n", line)
                letter_shape.append(line)

            print("word_shape印出來長這樣AAAA : \n", letter_shape)
            letter_shape = np.vstack(letter_shape)
            letter_shape = np.float32(letter_shape).reshape(-1, 3)
            print("word_shape印出來長這樣CCCCC : \n", letter_shape)
            # 3D -> 2D , change type to int32
            imgpts, jac = cv2.projectPoints(letter_shape, rvecs[i], tvecs[i], mtx, dist)
            imgpts = np.int32(imgpts)
            print("word_shape印出來長這樣DDDDD : \n", imgpts)
            # draw line
            for k, line_draw in enumerate(letter_shape[::2]):
                print(tuple(imgpts[k*2].ravel()), tuple(imgpts[k*2 + 1].ravel()))
                chessboard_with_word = cv2.line(cv2_image, tuple(imgpts[k*2].ravel()), tuple(imgpts[k*2 + 1].ravel()), (0, 0, 255), 5)
    

        # adjust window
        cv2.namedWindow("Chessboard with Word", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Chessboard with Word", 1200, 800)
        cv2.imshow("Chessboard with Word", chessboard_with_word)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()




def showOnVirtual(folder_path, input_text):

    if not folder_path.strip():
        print("Q2-2 Warning : 請選擇路徑")
        return True
    
    if input_text == 100:
        print("Q2-2 Warning : 請輸入文字")
        return True


        # set criteria
    chessboard_with_word = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(7,10,0)
    objp = np.zeros((11 * 8, 3), np.float32)
    for i in range(0, 8):
        for j in range(0, 11):
            objp[i * 11 + j] = [j, i, 0]

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3D point in real world space
    imgpoints = []  # 2D points in image plane

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    for image in image_files:
        image = folder_path + "/" + image
        image = cv2.imread(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (height, width), None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(corners2)

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)

    # create new_mtx
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
    anchor = [[7, 5, 0],[4, 5, 0],[1, 5, 0],[7, 2, 0],[4, 2, 0],[1, 2, 0]]
    
    # deal with every image
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    for i, image in enumerate(image_files):
        image = folder_path + "/" + image
        cv2_image = cv2.imread(image, cv2.COLOR_BGR2GRAY)
        
        # foreach input text
        for j, letter in enumerate(input_text):
            letter_shape = []
            latter_mat = loadAlphabetVirtical(folder_path, letter)
            
            # 每筆畫加上起始位子(讓每個字母在不同位子)
            for line in latter_mat:
                line += anchor[j]
                letter_shape.append(line)

            letter_shape = np.vstack(letter_shape)
            letter_shape = np.float32(letter_shape).reshape(-1, 3)

            # 3D -> 2D , change type to int32
            imgpts, jac = cv2.projectPoints(letter_shape, rvecs[i], tvecs[i], mtx, dist)
            imgpts = np.int32(imgpts)

            # draw line
            for k, line_draw in enumerate(letter_shape[::2]):
                chessboard_with_word = cv2.line(cv2_image, tuple(imgpts[k*2].ravel()), tuple(imgpts[k*2 + 1].ravel()), (0, 0, 255), 5)
    

        # adjust window
        cv2.namedWindow("Chessboard with Word", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Chessboard with Word", 1200, 800)
        cv2.imshow("Chessboard with Word", chessboard_with_word)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

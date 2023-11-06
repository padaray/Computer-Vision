import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

image_path_L_3 = ""
image_path_R_3 = ""

def stereoImageL():
    root = tk.Tk()
    root.withdraw()
    global image_path_L_3
    image_path_L_3 = filedialog.askopenfilename()

    if not image_path_L_3.strip():
        print("請選擇圖片")
        return True
    else:
        print("圖片L目前路徑：", image_path_L_3)
    
    # Load the image
    image = cv2.imread(image_path_L_3)

    # Display the image with keypoints
    cv2.namedWindow("stereoL", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("stereoL", 1200, 800)
    cv2.imshow('stereoL', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def stereoImageR():
    root = tk.Tk()
    root.withdraw()
    global image_path_R_3
    image_path_R_3 = filedialog.askopenfilename()

    if not image_path_R_3.strip():
        print("請選擇圖片")
        return True
    else:
        print("圖片R目前路徑：", image_path_R_3)
    
    # Load the image
    image = cv2.imread(image_path_R_3)

    # Display the image with keypoints
    cv2.namedWindow("stereoR", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("stereoR", 1200, 800)
    cv2.imshow('stereoR', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Q 3-1
def stereoDisparityMap():

    if not image_path_L_3.strip():
        print("請選擇圖片1")
        print("圖片1目前路徑：", image_path_L_3)
        return True
    if not image_path_R_3.strip():
        print("請選擇圖片2")
        print("圖片2目前路徑：", image_path_R_3)
        return True
    
    # imread two image
    left_image = cv2.imread(image_path_L_3, 0)
    right_image = cv2.imread(image_path_R_3, 0)

    # create stereo
    stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
    disparity = stereo.compute(left_image, right_image)

    # 視差值mapping到 0~255 灰階值範圍，為了可視化
    disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)

    # 視插圖轉為8 bit灰階圖
    disparity = disparity.astype(np.uint8)

    # save map
    map_savepath = image_path_L_3[:-7]
    map_savepath += "disparity_map.png"
    cv2.imwrite(map_savepath, disparity)

    # show image
    cv2.namedWindow("Disparity Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Disparity Map", 1400, 800)
    cv2.imshow('Disparity Map', disparity)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Q 3-2
def checkDisparity():

    # get disparity map path
    root = tk.Tk()
    root.withdraw()
    image_path_Disparity = filedialog.askopenfilename()

    if not image_path_L_3.strip():
        print("請選擇圖片左")
        print("圖片左目前路徑：", image_path_L_3)
        return True
    if not image_path_R_3.strip():
        print("請選擇圖片右")
        print("圖片右目前路徑：", image_path_R_3)
        return True
    if not image_path_Disparity.strip():
        print("請選擇圖片Map")
        print("圖片Map目前路徑：", image_path_Disparity)
        return True

    # read disparity map
    disparity_map = cv2.imread(image_path_Disparity, 0)

    # show left image
    left_image = cv2.imread(image_path_L_3)
    cv2.namedWindow("Left Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Left Image", 900, 600)
    cv2.namedWindow('Left Image')
    cv2.imshow('Left Image', left_image)

    # show right image
    right_image = cv2.imread(image_path_R_3)
    cv2.namedWindow("Right Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Right Image", 900, 600)
    cv2.namedWindow('Right Image')
    cv2.imshow('Right Image', right_image)

    # click event
    def on_left_image_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # get視差值
            disparity = disparity_map[y, x]
            
            # ignore視差值 = 0
            if disparity > 0:
                # 計算右圖座標
                right_x = x - disparity
                # draw point
                cv2.circle(right_image, (int(right_x), y), radius=5, color=(0, 255, 0),thickness=10)
                cv2.imshow('Right Image', right_image)

    # link click function to left image 
    cv2.setMouseCallback('Left Image', on_left_image_click)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
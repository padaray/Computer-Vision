import os
import tkinter as tk
from PIL import Image
from tkinter import filedialog
from hw1_ui import Ui_Dialog
from calibration import *
from augmented import *
from sift import *
from stereoDisplay import *
from vgg19 import *
from PyQt5 import QtCore, QtGui, QtWidgets


folder_path = " "
image_path_L = ""
image_path_R = ""

# Connect all button to function
class MyDialog(Ui_Dialog):
    def __init__(self, input_dia):
        super().__init__()

        Ui_Dialog.setupUi(self, input_dia)

        # Q1
        self.Load_Folder.clicked.connect(Load_Path.loadfolderPath)
        self.Find_Corner.clicked.connect(lambda:findCorner(folder_path))
        self.Find_Intrinsic.clicked.connect(lambda:findIntrMatrix(folder_path))
        self.Find_Extrinsic.clicked.connect(lambda:findExtrMatrix(folder_path, self.get_extrinsic_textbox()))
        self.Find_Distortion.clicked.connect(lambda:findDistMatrix(folder_path))
        self.Show_Undistorted_Result.clicked.connect(lambda:showUndistor(folder_path))

        # Q2
        self.Show_Word_On_Board.clicked.connect(lambda:showOnBoard(folder_path, self.get_showword_textbox()))
        self.Show_Word_Vertically.clicked.connect(lambda:showOnVirtual(folder_path, self.get_showword_textbox()))

        # Q3
        self.Load_Image_L.clicked.connect(lambda:stereoImageL())
        self.Load_Image_R.clicked.connect(lambda:stereoImageR())
        self.Stereo_Disparity_Map.clicked.connect(lambda:stereoDisparityMap())
        self.Check_Disparity_Value.clicked.connect(lambda:checkDisparity())

        # Q4
        self.Load_Image1.clicked.connect(Load_Path.loadImagePath_L)
        self.Load_Image2.clicked.connect(Load_Path.loadImagePath_R)
        self.Keypoints.clicked.connect(lambda:getKeyPoints(image_path_L))
        self.Matched_Keypoints.clicked.connect(lambda:getMacheKeypoints(image_path_L, image_path_R))

        # Q5
        self.Load_Image.clicked.connect(lambda:loadImage())
        self.Show_Augmented_Image.clicked.connect(lambda:showAugmentedImage())
        self.Show_Model_Structure.clicked.connect(lambda:showVGG19Strut())
        self.Show_Accuracy_and_Loss.clicked.connect(lambda:showAccAndLoss())
        self.Inference.clicked.connect(lambda:classifiImage())
    

    def get_extrinsic_textbox(self):
        value = self.Find_Extrinsic_Text.text()

        if not value.strip():
            return 100

        print(f"你選擇的是第{value}張圖\n")
        return int(value)
    
    def get_showword_textbox(self):
        value = self.Show_Word_Text.text()
        if not value.strip():
            return 100
        return value


# Load User folder path to folder_path
class Load_Path():
    def loadfolderPath(self):
        root = tk.Tk()
        root.withdraw()
        global folder_path
        folder_path = filedialog.askdirectory()

        print(folder_path)
        return folder_path

    # for Q 4-1
    def loadImagePath_L(self):
        root = tk.Tk()
        root.withdraw()
        global image_path_L
        image_path_L = filedialog.askopenfilename()

        print(image_path_L)
        return image_path_L

    # for Q 4-2
    def loadImagePath_R(self):
        root = tk.Tk()
        root.withdraw()
        global image_path_R
        image_path_R = filedialog.askopenfilename()

        print(image_path_R)
        return image_path_L
    
    
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = MyDialog(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
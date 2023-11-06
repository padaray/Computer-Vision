import cv2
import os
import numpy as np
import tkinter as tk
import torchsummary
import torch
import torch.nn as nn
from PIL import Image
from tkinter import filedialog
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optimizer
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

vgg_load_image_path = ""
folder_path = ""

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30)
])


# Q 5-0
def loadImage():
    root = tk.Tk()
    root.withdraw()
    global vgg_load_image_path
    vgg_load_image_path = filedialog.askopenfilename()
    vgg_load_image = cv2.imread(vgg_load_image_path)

    # Display the image
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 800, 600)
    cv2.imshow('image', vgg_load_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Q 5-1
def showAugmentedImage():

    if not vgg_load_image_path.strip():
        print("請選擇路徑")
        return True

    # get path
    root = tk.Tk()
    root.withdraw()
    global folder_path
    folder_path = filedialog.askdirectory()

    aug_images = []
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    # 先 resize 成一樣大小
    for i, image in enumerate(image_files):
        image = Image.open(folder_path + "/" + image)
        augmented_image = np.array(transform(image))
        augmented_image = cv2.resize(augmented_image, (300, 300))
        aug_images.append(augmented_image)

    canvas = np.zeros((900, 900, 3), dtype=np.uint8)
    for i in range(3):
        for j in range(3):
            canvas[i * 300:(i + 1) * 300, j * 300:(j + 1) * 300] = aug_images[i * 3 + j]

    # 显示九宫格图像
    cv2.imshow("Nine Augmented Images", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Q 5-2
def showVGG19Strut():
    # create VGG19 model
    model = models.vgg19(weights=False, num_classes=10)

    # print VGG19 structure
    torchsummary.summary(model, (3, 224, 224))


# Q 5-3
def showAccAndLoss():
    if not vgg_load_image_path.strip():
        print("請選擇路徑")
        return True

    # 使用字符串分割来获取斜杠分隔的部分
    parts = vgg_load_image_path.split('/')

    # 使用切片获取倒数第二个斜杠之前的部分
    acc_image_path = '/'.join(parts[:-2])

    acc_image_path = acc_image_path + "/Loss&Accu.png"

    print(acc_image_path)
    acc_image = cv2.imread(acc_image_path)

    # # Display the image
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 1500, 800)
    cv2.imshow('image', acc_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Q 5-4
def classifiImage():

    root = tk.Tk()
    root.withdraw()
    input_image = filedialog.askopenfilename()

    # show image
    show_image = cv2.imread(input_image)
    cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Input Image", 800, 600)
    cv2.imshow('Input Image', show_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --------------------to tensor---------------------
    # 读取图像
    image = Image.open(input_image)

    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为Tensor
    ])
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    # --------------------to tensor---------------------

    # 使用字符串分割来获取斜杠分隔的部分
    parts = input_image.split('/')

    # 使用切片获取倒数第二个斜杠之前的部分
    weight_path = '/'.join(parts[:-2])
    weight_path = weight_path + "/VGG19.pt"

    # load model
    model = models.vgg19_bn(weights=False, num_classes=10)
    checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()

    
    # 使用模型进行预测
    with torch.no_grad():
        output = model(image_tensor)

    # 获取模型的预测概率分布
    probabilities = torch.softmax(output, dim=1)
    
    probabilities = probabilities[0].numpy()

    class_names = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # 显示预测概率分布
    plt.bar(class_names, probabilities)
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.show()
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

def SIFT():
    imgPath = '/home/mu/PycharmProjects/sift/.venv/res/6c140c54-a249-4e36-b44f-a08ad90844bf.webp'
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    if imgGray is None:
        print(f"Error: Unable to read image at {imgPath}")
        return

    sift = cv.SIFT_create()
    keypoints = sift.detect(imgGray, None)
    imgGray = cv.drawKeypoints(imgGray, keypoints, imgGray, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    resultPath = '/home/mu/PycharmProjects/sift/.venv/res/result.webp'
    success = cv.imwrite(resultPath, imgGray)

    if not success:
        print(f"Error: Unable to save image at {resultPath}")
        return

    print(f"Image saved successfully at {resultPath}")

    plt.figure()
    plt.imshow(imgGray)
    plt.show()

SIFT()
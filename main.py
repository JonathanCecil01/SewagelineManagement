
import cv2
import os
import matplotlib.pyplot as plt
import skimage
import numpy as np
import PIL
from PIL import Image
import math
import time

#Converting image to an Numpy Array
def loadconvert(link):
    image = cv2.imread(link)
    data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return data

#Segmentation of the image to retrieve the pipes based on color
def segmentation(data):
    hsv_data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)
    lower = np.array([28, 210, 210], dtype="uint8")
    upper = np.array([70, 255, 255], dtype="uint8")
    mask = cv2.inRange(hsv_data, lower, upper)
    result = cv2.bitwise_and(data, data, mask =mask)
    return result

#Line detection using Hough lines
def lineDetection(data):
    print(data.size)
    #conver to gray scale
    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    low_threshold = 50
    high_threshold = 150
    kernel_size = 5
    lines_list = []
    #Gaussian Blur to reduce pixels
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    # Line Detection with Hough Lines Algorithm
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=15, maxLineGap=8)  # 15, 8
    # Assigning Lines
    for points in lines:
        x1, y1, x2, y2 = points[0]
        #cv2.line(data, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lines_list.append([x1, y1, x2, y2])
    # plt.imshow(data) #Uncomment to see the crude segments (Change Parameters in HoughLines to get a lower count )
    # plt.show()
    return lines


#refining the lines
def refineLine(lines, data):
    minimum =200
    sum =0
    count = 0
    for points in lines:
        if points[0][0] == 0 and points[0][3]==0:
            continue
        slope = (points[0][3] - points[0][1]) / (points[0][2] - points[0][0])
        slope = round(slope, 2)
        for spoints in lines:
            if points[0][0] == spoints[0][0] and points[0][1] == spoints[0][1] and points[0][2] == spoints[0][2] and \
                    points[0][3] == spoints[0][3]:
                continue
            else:
                if (spoints[0][0] == 0 and spoints[0][1] == 0):
                    continue
                else:
                    p2 = [points[0][2], points[0][3]]
                    q1 = [spoints[0][0], spoints[0][1]]
                    p1 = [points[0][0], points[0][1]]
                    q2 = [spoints[0][2], spoints[0][3]]
                    slope2 = (spoints[0][3] - spoints[0][1]) / (spoints[0][2] - spoints[0][0])
                    slope2 = round(slope2, 2)
                    c1 = (points[0][3] - points[0][1]) * points[0][0] + (points[0][0] - points[0][2]) * points[0][1]
                    c2 = (spoints[0][3] - spoints[0][1]) * spoints[0][0] + (spoints[0][0] - spoints[0][2]) * spoints[0][
                        1]
                    # print("Intercept 1", c1, "Intercept 2", c2)
                    # print("Slope 1 ", slope, "Slope 2 ", slope2)
                    if abs(c1 - c2) < minimum:
                        minimum = abs(c1 - c2)
                    if abs(slope - slope2) == 0 and abs(c1 - c2) <= 800:
                        sum += abs(c1 - c2)
                        # print(abs(c1 - c2))
                        count += 1
                        spoints[0] = [0, 0, 0, 0]

    lines_list = []
    for points in lines:
        x1, y1, x2, y2 = points[0]
        if x1 == 0 and x2 == 0 and y1 == 0 and y2 == 0:
            continue
        else:
            #cv2.line(data, (x1, y1), (x2, y2), (0, 0, 255), 2)
            lines_list.append([x1, y1, x2, y2])
    #print(len(lines_list)) #Uncomment to check the number of lines
    # plt.imshow(data) #Uncomment to View the Final Plot
    # plt.show()
    return lines_list

#plotting the lines
def plotlines(lines_list, data):
    for points in lines_list:
        cv2.line(data, (points[0], points[1]), (points[2], points[3]), (255,0,0), 2)
    plt.imshow(data)
    plt.show()
link = '/Users/jonathancecil/Downloads/SantaCruzSewage.jpeg'
data = loadconvert(link)
data = segmentation(data)
lines = lineDetection(data)
lines_list = refineLine(lines, data)
plotlines(lines_list, data)


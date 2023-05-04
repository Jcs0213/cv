import cv2
import numpy as np

# 读取图片
img = cv2.imread('./dataset/2.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,sigmaX=3,sigmaY=3,ksize=(0,0))
thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 找到图像中的所有轮廓
contours, hierarchy = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

use_contours=[]
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if h!=0:
        if w/h >9:
            use_contours.append(contour)

for contour in use_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 可视化结果并输出
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

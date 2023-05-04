import cv2
import numpy as np

# 读取图片
img = cv2.imread('4.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, sigmaX=3, sigmaY=3, ksize=(0, 0))
thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

sobelx = cv2.Sobel(closed, cv2.CV_64F, 0, 1, ksize=3)
abs_sobelx = np.absolute(sobelx)
sobel_8u = np.uint8(abs_sobelx)

# 霍夫变换检测直线
lines = cv2.HoughLinesP(sobel_8u, rho=1, theta=np.pi / 180, threshold=20, minLineLength=50, maxLineGap=1)
# center = (img.shape[1]//2, img.shape[0]//2)
# distances = []
# 找到最长的四根线
# 将所有直线按照斜率排序
lines = sorted(lines, key=lambda x: np.arctan2(x[0][3] - x[0][1], x[0][2] - x[0][0]))

sorted_lines = sorted(lines, key=lambda x: np.sqrt((x[0][2] - x[0][0]) ** 2 + (x[0][3] - x[0][1]) ** 2), reverse=True)[
               :4]

for line in lines:
    cv2.line(img, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 255), 3)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

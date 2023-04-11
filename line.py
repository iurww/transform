import cv2
from PIL import Image
import numpy as np
import random

img = cv2.imread("./s_img/3.jpg", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 240, 440)
# bw_image = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite("./file/line.jpg", edges)



sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)

magnitude = np.sqrt(sobelx**2 + sobely**2)
angle = np.arctan2(sobely, sobelx)

angle = (angle + np.pi) * 180 / np.pi

line_angles = np.zeros_like(edges)

for y in range(1, edges.shape[0]-1):
    for x in range(1, edges.shape[1]-1):
        if edges[y][x] != 0:
            angle_slice = angle[y-1:y+2, x-1:x+2]
            angle_median = np.median(angle_slice)
            line_angles[y][x] = angle_median


# for i in range(line_angles.shape[0]):
#     for j in range(line_angles.shape[1]):
#         if line_angles[i][j] > 0:
#             print(i,j,line_angles[i][j])
    
# cv2.imshow("line angles", line_angles)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = Image.open("line.jpg")
# img = np.asarray(img).astype(np.float32) / 255.0




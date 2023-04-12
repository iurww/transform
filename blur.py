import cv2
from PIL import Image
import numpy as np

image = cv2.imread("./file/style.jpg")
grid_size=(10, 10)

# 获取图片大小
height, width, _ = image.shape

# 计算每个网格的大小
grid_height = height // grid_size[0]
grid_width = width // grid_size[1]

# 切割图片为网格
grids = []
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        y1, x1 = i * grid_height, j * grid_width
        y2, x2 = (i + 1) * grid_height, (j + 1) * grid_width
        grid = image[y1:y2, x1:x2]
        grids.append(grid)

# 对每个网格的边缘进行处理
for i, grid in enumerate(grids):
    if i % grid_size[1] != 0:  # 左边界处理
        left_edge = (grid[:, 0, :] + grids[i-1][:, -1, :]) / 2
        grid[:, 0, :] = left_edge

    if (i + 1) % grid_size[1] != 0:  # 右边界处理
        right_edge = (grid[:, -1, :] + grids[i+1][:, 0, :]) / 2
        grid[:, -1, :] = right_edge

    if i // grid_size[1] != 0:  # 上边界处理
        top_edge = (grid[0, :, :] + grids[i-grid_size[1]][-1, :, :]) / 2
        grid[0, :, :] = top_edge

    if i // grid_size[1] != grid_size[0] - 1:  # 下边界处理
        bottom_edge = (grid[-1, :, :] + grids[i+grid_size[1]][0, :, :]) / 2
        grid[-1, :, :] = bottom_edge

# 将处理好的网格拼接回去
output = np.concatenate([np.concatenate(grids[i:i+grid_size[1]], axis=1) for i in range(0, len(grids), grid_size[1])], axis=0)
cv2.imwrite("./file/blur.jpg", output)

# image = cv2.imread('./file/output.jpg')
# blur = cv2.bilateralFilter(image, 4, 1000, 1000)
# cv2.imwrite('./file/blur_image.jpg', blur)


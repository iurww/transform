import cv2

# 读取图像
img = cv2.imread('./style_image/2.jpg')

# 图像分割
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200)

# 查找轮廓
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍历轮廓，提取笔触
for i, contour in enumerate(contours):
    # 计算轮廓周长
    perimeter = cv2.arcLength(contour, True)
    # 根据周长进行筛选，只保留周长在一定范围内的轮廓
    if perimeter > 100 and perimeter < 1000:
        # 绘制轮廓
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
        # 保存笔触图像
        x, y, w, h = cv2.boundingRect(contour)
        stroke = img[y:y+h, x:x+w]
        cv2.imwrite('./stroke/stroke_' + str(i) + '.jpg', stroke)

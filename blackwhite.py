import cv2
from PIL import Image, ImageChops


img = cv2.imread('./c_img/2.jpg', cv2.IMREAD_GRAYSCALE)
thresh, new_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

print(thresh)
cv2.imwrite('./file/bw.jpg', new_img)

img1 = Image.open('./file/bw.jpg')
width,height=img1.size
print(width, height)

px = img1.load()
w, h = img1.size
for i in range(w):
    for j in range(h):
        if type(px[i, j]) == int:
            px[i, j] = 255-px[i, j]
        elif len(px[i, j]) == 3:
                px[i, j] = tuple([255-i for i in px[i, j]])
        elif len(px[i, j]) == 4:
                px[i, j] = tuple([255-i for i in px[i, j][:3]]+[px[i, j][-1]])
        else:
            pass
img1.save("./file/wb.jpg")

# img = cv2.imread('./file/style.jpg')

# blur_img = cv2.GaussianBlur(img, (9, 9), 0)

# cv2.imwrite('output.jpg', blur_img)
# cv2.waitKey()

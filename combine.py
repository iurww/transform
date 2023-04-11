from PIL import Image
import random

def combine_image(image):
    joint = Image.new("RGB", (512, 512))
    # mask = Image.new('L', (512, 512), 300)
    joint.paste(image.resize((256,256)),(0, 0))
    joint.paste(image.resize((256,256)),(256,0))

    for i in range(4):
        for j in range(2,4):
            joint.paste(image.rotate(random.randint(0,0)*90),(i*128, j*128))
    return joint

# 读取单个图片
image = Image.open('./t_img/1.jpg')
# 随机旋转图片并拼接
combine_image(image).save("./file/combine.jpg")
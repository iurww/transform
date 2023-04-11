from PIL import Image, ImageDraw
import random

brush_image = Image.open("./t_img/2.jpg").convert("RGBA")

canvas_size = 1280
canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
canvas.alpha_composite(brush_image.resize((1280,1280)), dest=(0,0))

num_brushes = 1000

for i in range(num_brushes):
    angle = random.randint(0,359) 
    scale = random.uniform(1.2, 1.7)
    brush_transformed = brush_image.rotate(angle).resize((int(128*scale), int(128*scale)), resample=Image.Resampling.BOX)

    alpha = brush_transformed.split()[3]
    bbox = alpha.getbbox()
    if bbox:
        brush_transformed = brush_transformed.crop(bbox)

    canvas_width, canvas_height = canvas.size
    brush_width, brush_height = brush_transformed.size
    x = random.randint(0, canvas_width - brush_width)
    y = random.randint(0, canvas_height - brush_height)
    position = (x, y)
    canvas.alpha_composite(brush_transformed, dest=position)

canvas.save("./file/texture.png")

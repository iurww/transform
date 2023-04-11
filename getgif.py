from PIL import Image
import glob
 
image_path_list = sorted(glob.glob('./t_img/*.jpg'))
 
frames = []
for image_path in image_path_list:
    image = Image.open(image_path)
    frames.append(image)
 
frames[0].save('./file/out.gif', save_all=True, append_images=frames[1:], duration=200, loop=0)

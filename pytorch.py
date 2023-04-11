import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = transforms.Compose([
    # transforms.Resize(128),
    transforms.ToTensor()])
unloader = transforms.ToPILImage()

style_image_path = "./s_img/6.jpg"
content_image_path = "./c_img/2.jpg"
output_image_path = "./file/output.jpg"

def load_image(fp, need_resize=True):
    image = Image.open(fp)
    if need_resize:
        image = image.resize((256, 256), box=None, reducing_gap=None)
    # image.show()
    image = loader(image).unsqueeze(0)
    image = image.to(device, torch.float)
    return image

def save_image(fp, image):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.show()
    image.save(fp)

def gram(feature):
    a, b, c, d = feature.size()
    feature = feature.view(a * b, c * d)
    gram = torch.mm(feature, feature.t())
    return gram.div(a * b * c * d)


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = 0

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram(target_feature).detach()
        self.loss = 0

    def forward(self, input):
        G = gram(input)
        self.loss = F.mse_loss(G, self.target)
        return input


if __name__ == "__main__":
    style_image = load_image(style_image_path, True)
    content_image = load_image(content_image_path, True)
    input_image = content_image.clone()
    # input_image = torch.randn(content_image.size(), device=device)
    print(device)

    cnn = models.vgg19(pretrained=True)
    pre_file = torch.load('C:\\Users\\wwr\\.cache\\torch\\hub\\checkpoints\\vgg19-dcbb9e9d.pth')
    cnn.load_state_dict(pre_file)
    cnn = cnn.features.eval()
   
    style_losses = []
    content_losses = []
    s_img = style_image.detach()
    c_img = content_image.detach()
    print("style image size :", s_img.size())
    print("content image size :", c_img.size())

    for i in range(37):
        s_img = cnn[i](s_img)
        c_img = cnn[i](c_img)
        # print("s_img:", s_img.size())
        # print("c_img:", c_img.size())
        if i == 1 or i == 6 or i == 11 or i == 20 or i == 29:
            sl = StyleLoss(s_img.detach())
            style_losses.append(sl)
        if i == 22:
            cl = ContentLoss(c_img.detach())
            content_losses.append(cl)

    model = nn.Sequential()
    for i in range(30):
        model.add_module(str(i), cnn[i])
        if (i == 1):
            model.add_module("sl1", style_losses[0])
        if (i == 6):
            model.add_module("sl2", style_losses[1])
        if (i == 11):
            model.add_module("sl3", style_losses[2])
        if (i == 20):
            model.add_module("sl4", style_losses[3])
        if (i == 29):
            model.add_module("sl5", style_losses[4])
        if (i == 22):
            model.add_module("cl1", content_losses[0])

    optimizer = optim.LBFGS([input_image.requires_grad_()])
    for epoch in range(20):
        def closure():
            input_image.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_image)
            style_loss, content_loss = 0, 0
            style_loss += style_losses[0].loss * 3
            for sl in style_losses:
                style_loss += sl.loss
            for cl in content_losses:
                content_loss += cl.loss
            loss = style_loss * 10000000 + content_loss
            loss.backward()
            print(epoch, style_loss.item(), content_loss.item(), loss.item())
            return loss.item()


        optimizer.step(closure)
        save_image(output_image_path, input_image)

    save_image(output_image_path, input_image)

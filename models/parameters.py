import torch
from torchsummary import summary
from resnet import resnet8
from vgg import vgg19, vgg16, vgg13, vgg11, vgg8
from mobilenetv2 import MobileNetV2

model = MobileNetV2(num_classes=100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

summary(model.to(device), input_size=(3, 32, 32))
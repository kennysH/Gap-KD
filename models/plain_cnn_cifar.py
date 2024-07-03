import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as torch_models
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary


class ConvNetMaker(nn.Module):
    """
    Creates a simple (plane) convolutional neural network
    """
    def __init__(self, layers):
        """
        Makes a cnn using the provided list of layers specification
        The details of this list is available in the paper
        :param layers: a list of strings, representing layers like ["CB32", "CB32", "FC10"]
        """
        super(ConvNetMaker, self).__init__()
        self.conv_layers = []
        self.fc_layers = []
        h, w, d = 32, 32, 3
        previous_layer_filter_count = 3
        previous_layer_size = h * w * d
        num_fc_layers_remained = len([1 for l in layers if l.startswith('FC')])
        for index, layer in enumerate(layers):
            if layer.startswith('Conv'):
                filter_count = int(layer[4:])
                self.conv_layers += [nn.Conv2d(previous_layer_filter_count, filter_count, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(filter_count), nn.ReLU(inplace=True)]
                previous_layer_filter_count = filter_count
                d = filter_count
                previous_layer_size = h * w * d
            elif layer.startswith('MaxPool'):
                self.conv_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                h, w = int(h / 2.0), int(w / 2.0)
                previous_layer_size = h * w * d
            elif layer.startswith('FC'):
                num_fc_layers_remained -= 1
                current_layer_size = int(layer[2:])
                if num_fc_layers_remained == 0:
                    self.fc_layers += [nn.Linear(previous_layer_size, current_layer_size)]
                else:
                    self.fc_layers += [nn.Linear(previous_layer_size, current_layer_size), nn.ReLU(inplace=True)]
                previous_layer_size = current_layer_size

        conv_layers = self.conv_layers
        fc_layers = self.fc_layers
        self.conv_layers = nn.Sequential(*conv_layers)
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x



plane_cifar10_book = {
    '2': ['Conv16', 'MaxPool', 'Conv16', 'MaxPool', 'FC10'],
    '3': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'MaxPool', 'FC100'],
    '4': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'FC10'],
    '5': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'FC100'],
    '6': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'FC10'],
    '7': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool', 'Conv256', 'MaxPool', 'FC64', 'FC100'],
    '8': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128','MaxPool', 'FC64', 'FC10'],
    '9': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool', 'Conv256', 'Conv256', 'Conv256', 'MaxPool', 'FC512', 'FC100'],
    '10': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool', 'Conv256', 'Conv256', 'Conv256', 'Conv256', 'MaxPool', 'FC128','FC10'],
}


plane_cifar100_book = {
    '2': ['Conv32', 'MaxPool', 'Conv32', 'MaxPool', 'FC100'],
    '3': ['Conv32', 'Conv32', 'MaxPool', 'Conv64',  'MaxPool', 'FC100'],
    '4': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'FC100'],
    '5': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'FC100'],
    '6': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool','Conv128', 'Conv128' ,'FC100'],
    '7': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool', 'Conv256', 'MaxPool', 'FC64', 'FC100'],
    '8': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool', 'Conv256', 'Conv256','MaxPool', 'FC64', 'FC100'],
    '9': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool', 'Conv256', 'Conv256', 'Conv256', 'MaxPool', 'FC512', 'FC100'],
    '10': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool', 'Conv256', 'Conv256', 'Conv256', 'Conv256', 'MaxPool', 'FC512', 'FC100'],
}



def create_cnn_model(name, num_classes, use_cuda=False):
    """
    Create a student for training, given student name and dataset
    :param name: name of the student. e.g., resnet110, resnet32, plane2, plane10, ...
    :param dataset: the dataset which is used to determine last layer's output size. Options are cifar10 and cifar100.
    :return: a pytorch student for neural network
    """
    model = None
    plane_size = name[5:]
    model_spec = plane_cifar10_book.get(plane_size) if num_classes == 10 else plane_cifar100_book.get(plane_size)
    plane_model = ConvNetMaker(model_spec)
    model = plane_model

    # copy to cuda if activated
    if use_cuda:
        model = model.cuda()

    return model

def plane2(num_classes, use_cuda=False):
    return create_cnn_model("plane2", num_classes, use_cuda)

def plane4(num_classes, use_cuda=False):
    return create_cnn_model("plane4", num_classes, use_cuda)

def plane6(num_classes, use_cuda=False):
    return create_cnn_model("plane6", num_classes, use_cuda)

def plane8(num_classes, use_cuda=False):
    return create_cnn_model("plane8", num_classes, use_cuda)

def plane10(num_classes, use_cuda=False):
    return create_cnn_model("plane10", num_classes, use_cuda)

# if __name__ == "__main__":
    # dataset = 'cifar100'
    # print('planes')
    # for p in [2, 4, 6, 8, 10]:
    # 	plane_name = "plane" + str(p)
    # 	print(create_cnn_model(plane_name, dataset))

# import torch.nn as nn
# from torchsummary import summary
# import torch.nn.functional as F
# class Plane2(nn.Module):
#     def __init__(self, num_classes=100):
#         super(Plane2, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Linear(32 * 8 * 8, num_classes)
#         )

#     def forward(self, x, is_feat=False, preact=False):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)  # flatten the tensor
#         x = self.fc_layers(x)
#         return x
    
# class Plane4(nn.Module):
#     def __init__(self, num_classes=100):
#         super(Plane4, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Linear(64 * 8 * 8, num_classes)
#         )

#     def forward(self, x, is_feat=False, preact=False):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)  # flatten the tensor
#         x = self.fc_layers(x)
#         return x
    
# class Plane6(nn.Module):
#     def __init__(self, num_classes=100):
#         super(Plane6, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True)
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Linear(8192, num_classes)
#         )

#     def forward(self, x, is_feat=False, preact=False):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)  # flatten the tensor
#         x = self.fc_layers(x)
#         return x
    

# class Plane8(nn.Module):
#     def __init__(self, num_classes=100):
#         super(Plane8, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Linear(256 * 2 * 2, 64),
#             nn.ReLU(inplace=True),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, x, is_feat=False, preact=False):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)  # flatten the tensor
#         x = self.fc_layers(x)
#         return x

# class Plane10(nn.Module):
#     def __init__(self, num_classes=100):
#         super(Plane10, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Linear(256 * 2 * 2, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x, is_feat=False, preact=False):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)  # flatten the tensor
#         x = self.fc_layers(x)
#         return x
    
# def plane2(**kwargs):
#     return Plane2(**kwargs)

# def plane4(**kwargs):
#     return Plane4(**kwargs)

# def plane6(**kwargs):
#     return Plane6(**kwargs)

# def plane8(**kwargs):    
#     return Plane8(**kwargs)

# def plane10(**kwargs):
#     return Plane10(**kwargs)   

if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    net = plane10(num_classes=100)
    logit = net(x)
    print(logit.shape)
     # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将模型移动到正确的设备
    net = net.to(device)

    # 计算模型的参数
    summary(net.to(device), input_size=(3, 32, 32))


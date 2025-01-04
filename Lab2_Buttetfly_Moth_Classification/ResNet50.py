import torch
import torch.nn as nn
import torchvision

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, id_layer, is_first=False):
        super(Bottleneck, self).__init__()
            
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        if is_first and id_layer != 1: #stride according to id_layer
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False) 
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        if is_first:
            if id_layer != 1:
                self.down_sample = nn.Sequential(
                    nn.Conv2d(out_channels*2, out_channels*4, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(out_channels*4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
                )
            else:
                self.down_sample = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_channels*4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
                )
        else:
            self.down_sample = None
        
    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.down_sample is not None:
            identity = self.down_sample(identity)
        
        x += identity
        x = self.relu(x)

        return x
        
    
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.layer1 = nn.Sequential(
            Bottleneck(in_channels=64, out_channels=64, id_layer=1, is_first=True),
            Bottleneck(in_channels=256, out_channels=64, id_layer=1, is_first=False),
            Bottleneck(in_channels=256, out_channels=64, id_layer=1, is_first=False)
        )
        self.layer2 = nn.Sequential(
            Bottleneck(in_channels=256, out_channels=128, id_layer=2, is_first=True),
            Bottleneck(in_channels=512, out_channels=128, id_layer=2, is_first=False),
            Bottleneck(in_channels=512, out_channels=128, id_layer=2, is_first=False),
            Bottleneck(in_channels=512, out_channels=128, id_layer=2, is_first=False)
        )
        self.layer3 = nn.Sequential(
            Bottleneck(in_channels=512, out_channels=256, id_layer=3, is_first=True),
            Bottleneck(in_channels=1024, out_channels=256, id_layer=3, is_first=False),
            Bottleneck(in_channels=1024, out_channels=256, id_layer=3, is_first=False),
            Bottleneck(in_channels=1024, out_channels=256, id_layer=3, is_first=False),
            Bottleneck(in_channels=1024, out_channels=256, id_layer=3, is_first=False),
            Bottleneck(in_channels=1024, out_channels=256, id_layer=3, is_first=False)
        )
        self.layer4 = nn.Sequential(
            Bottleneck(in_channels=1024, out_channels=512, id_layer=4, is_first=True),
            Bottleneck(in_channels=2048, out_channels=512, id_layer=4, is_first=False),
            Bottleneck(in_channels=2048, out_channels=512, id_layer=4, is_first=False)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(in_features=2048, out_features=100, bias=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x



# pytorch_resnet50 = torchvision.models.resnet50(weights=None)
# print(pytorch_resnet50)

# my_resnet50 = ResNet50()
# print(my_resnet50)


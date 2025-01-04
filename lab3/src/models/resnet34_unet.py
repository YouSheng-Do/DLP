# Implement your ResNet34_UNet model here
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, id_layer, is_first=False):
        super(Bottleneck, self).__init__()
            
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        if is_first and id_layer != 1: #stride according to id_layer
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False) 
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        if is_first and id_layer != 1:
                self.down_sample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
                )
        else:
            self.down_sample = None
        
    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.down_sample is not None:
            identity = self.down_sample(identity)
        
        x += identity
        x = self.relu(x)

        return x
        
    
class ResNet34Encoder(nn.Module):
    def __init__(self):
        super(ResNet34Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.layer1 = nn.Sequential(
            Bottleneck(in_channels=64, out_channels=64, id_layer=1, is_first=True),
            Bottleneck(in_channels=64, out_channels=64, id_layer=1, is_first=False),
            Bottleneck(in_channels=64, out_channels=64, id_layer=1, is_first=False)
        )
        self.layer2 = nn.Sequential(
            Bottleneck(in_channels=64, out_channels=128, id_layer=2, is_first=True),
            Bottleneck(in_channels=128, out_channels=128, id_layer=2, is_first=False),
            Bottleneck(in_channels=128, out_channels=128, id_layer=2, is_first=False),
            Bottleneck(in_channels=128, out_channels=128, id_layer=2, is_first=False)
        )
        self.layer3 = nn.Sequential(
            Bottleneck(in_channels=128, out_channels=256, id_layer=3, is_first=True),
            Bottleneck(in_channels=256, out_channels=256, id_layer=3, is_first=False),
            Bottleneck(in_channels=256, out_channels=256, id_layer=3, is_first=False),
            Bottleneck(in_channels=256, out_channels=256, id_layer=3, is_first=False),
            Bottleneck(in_channels=256, out_channels=256, id_layer=3, is_first=False),
            Bottleneck(in_channels=256, out_channels=256, id_layer=3, is_first=False)
        )
        self.layer4 = nn.Sequential(
            Bottleneck(in_channels=256, out_channels=512, id_layer=4, is_first=True),
            Bottleneck(in_channels=512, out_channels=512, id_layer=4, is_first=False),
            Bottleneck(in_channels=512, out_channels=512, id_layer=4, is_first=False)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        return x1, x2, x3, x4, x5    

class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1x1 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)  
        self.decode = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.up(x)
        x = self.conv1x1(x)
        x = self.decode(x)

        return x

class UNetDecoder(nn.Module):
    def __init__(self, num_classes):
        super(UNetDecoder, self).__init__()
        
        self.decode1 = DecodeBlock(256 + 512, 32)
        self.decode2 = DecodeBlock(32 + 256, 32)
        self.decode3 = DecodeBlock(32 + 128, 32)
        self.decode4 = DecodeBlock(32 + 64, 32)
        self.decode5 = nn.Sequential(
            nn.ConvTranspose2d(32, 32 // 2, kernel_size=2, stride=2),
            nn.Conv2d(32 // 2, 32, kernel_size=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x, encoder_outputs):
        x1, x2, x3, x4 = encoder_outputs
        x = self.decode1(x, x4)
        x = self.decode2(x, x3)
        x = self.decode3(x, x2)
        x = self.decode4(x, x1)
        x = self.decode5(x)
        
        x = self.conv(x)
        
        return x


class ResNet34_UNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34_UNet, self).__init__()

        self.encoder = ResNet34Encoder()
        self.decoder = UNetDecoder(num_classes)

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        x5 = encoder_outputs[-1]
        
        x = self.decoder(x5, encoder_outputs[:-1])
        # print(x.shape)
        return x




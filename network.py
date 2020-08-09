import torch.nn as nn
import torch.nn.functional as F
import torch

class Encoder(nn.Module):
    def __init__(self): #model_dict=None):
        super(Encoder, self).__init__()

        from torchvision import models
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.vgg16_bn(pretrained=False).features
        
    def forward(self, xb):
        xb = self.model(xb)
        
        return xb
        

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.bn7_3 = nn.BatchNorm2d(num_features=512)
        self.bn8_1 = nn.BatchNorm2d(num_features=256)
        self.bn8_2 = nn.BatchNorm2d(num_features=256)
        self.bn9_1 = nn.BatchNorm2d(num_features=128)
        self.bn9_2 = nn.BatchNorm2d(num_features=128)
        self.bn10_1 = nn.BatchNorm2d(num_features=64)
        self.bn10_2 = nn.BatchNorm2d(num_features=64)
        self.drop_layer = nn.Dropout2d(p=0.5)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', 
                                    align_corners=False)
        self.conv7_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv8_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv9_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv9_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv10_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv10_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        
    def forward(self, xb):
        xb = F.relu(self.bn7_3(self.conv7_3(xb)))
        xb = F.relu(self.bn8_1(self.conv8_1(xb)))
        xb = F.relu(self.bn8_2(self.conv8_2(xb)))
        xb = self.upsample(xb)
        xb = F.relu(self.bn9_1(self.conv9_1(xb)))
        xb = F.relu(self.bn9_2(self.conv9_2(xb)))
        xb = self.upsample(xb)
        xb = F.relu(self.bn10_1(self.conv10_1(xb)))
        xb = F.relu(self.bn10_2(self.conv10_2(xb)))
        return self.output(xb)
        
        
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, inp):
        inp = self.encoder(inp)
        inp = self.decoder(inp)
        return inp
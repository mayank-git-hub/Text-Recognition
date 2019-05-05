import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Inconv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        self.bilinear = bilinear 
        if not bilinear:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)            

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):

        if not self.bilinear:
            x1 = self.up(x1)
        else:
            x1 = nn.functional.interpolate(input=x1, scale_factor=2 , mode='bilinear', align_corners=True)

        diffx = x1.size()[2] - x2.size()[2]
        diffy = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffx // 2, int(diffx / 2),
                        diffy // 2, int(diffy / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class JustUp(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(JustUp, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1):
        x1 = self.up(x1)
        x1 = self.conv(x1)
        return x1


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

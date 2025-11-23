import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Standard Convolution Block: Conv2d + BatchNorm2d + SiLU Activation
    """
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """
    Standard YOLOv5 Bottleneck
    """
    def __init__(self, c1, c2, shortcut=True, e=0.5): # e = expansion factor
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBlock(c1, c_, k=1, s=1, p=0)
        self.cv2 = ConvBlock(c_, c2, k=3, s=1, p=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    """
    Simplified C3 Block (CSP Bottleneck with 3 convolutions)
    """
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBlock(c1, c_, k=1, s=1, p=0)
        self.cv2 = ConvBlock(c1, c_, k=1, s=1, p=0)
        self.cv3 = ConvBlock(2 * c_, c2, k=1, s=1, p=0)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF)
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvBlock(c1, c_, k=1, s=1, p=0)
        self.cv2 = ConvBlock(c_ * 4, c2, k=1, s=1, p=0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with torch.no_grad():
            y1 = self.m(x)
            y2 = self.m(y1)
            y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class YOLOv5Scratch(nn.Module):
    """
    A simplified 'from scratch' YOLOv5-like model.
    """
    def __init__(self, num_classes=80):
        super().__init__()
        # Simple Backbone
        self.layer1 = ConvBlock(3, 32, k=6, s=2, p=2)  # Stem (replaces Focus)
        self.layer2 = ConvBlock(32, 64, k=3, s=2, p=1)
        self.layer3 = C3(64, 64, n=1)
        self.layer4 = ConvBlock(64, 128, k=3, s=2, p=1)
        self.layer5 = C3(128, 128, n=2)
        self.layer6 = ConvBlock(128, 256, k=3, s=2, p=1)
        self.layer7 = C3(256, 256, n=1)
        self.layer8 = SPPF(256, 256)

        # Simple Head (to match your 'YOLOv12' example)
        # This is NOT a real PANet neck, but it fits your simple setup.
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.SiLU(),
            # 3 anchors, (x, y, w, h, conf + num_classes)
            nn.Conv2d(128, (num_classes + 5) * 3, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return self.head(x)
import torch
import torch.nn as nn
from typing import List

# --- Re-using your exact building blocks ---

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

class C2f(nn.Module):
    """
    Modern C2f Block (from YOLOv8)
    """
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        self.c = int(c2 * e) 
        self.cv1 = ConvBlock(c1, 2 * self.c, k=1, s=1, p=0)
        self.cv2 = ConvBlock((2 + n) * self.c, c2, k=1, s=1, p=0)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n))

    def forward(self, x):
        y = self.cv1(x)
        y1, y2 = y.split((self.c, self.c), 1)
        fused_features: List[torch.Tensor] = [y1, y2]
        for m in self.m:
            y2 = m(y2)
            fused_features.append(y2)
        return self.cv2(torch.cat(fused_features, 1))

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
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class SimpleDecoupledHead(nn.Module):
    """
    A simplified, anchor-free, decoupled head.
    (Re-using the head from YOLOModernScratch)
    """
    def __init__(self, in_channels, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        c_ = 128 # Fixed hidden channels
        
        self.common_conv = ConvBlock(in_channels, c_, k=1, s=1, p=0)
        self.reg_head = nn.Conv2d(c_, 4, 1)
        self.cls_head = nn.Conv2d(c_, num_classes, 1)

    def forward(self, x):
        x = self.common_conv(x)
        reg_output = self.reg_head(x)
        cls_output = self.cls_head(x)
        return torch.cat([reg_output, cls_output], dim=1)


# --- NEW YOLOv8 Model ---

class YOLOv8Scratch(nn.Module):
    """
    A simplified 'from scratch' model of YOLOv8.
    
    This implementation adds the PANet neck and multi-scale
    decoupled heads to the 'YOLOModernScratch' design.
    """
    def __init__(self, num_classes=80):
        super().__init__()
        
        # --- 1. Backbone ---
        # Using the exact backbone from your YOLOModernScratch
        # It produces feature maps at 3 scales:
        # P3: /4 scale (from layer3, 64 channels)
        # P4: /8 scale (from layer5, 128 channels)
        # P5: /16 scale (from layer8, 256 channels)
        self.layer1 = ConvBlock(3, 32, k=6, s=2, p=2)   # /2
        self.layer2 = ConvBlock(32, 64, k=3, s=2, p=1)  # /4
        self.layer3 = C2f(64, 64, n=1, shortcut=True)   # P3 out (64 ch)
        
        self.layer4 = ConvBlock(64, 128, k=3, s=2, p=1) # /8
        self.layer5 = C2f(128, 128, n=2, shortcut=True) # P4 out (128 ch)
        
        self.layer6 = ConvBlock(128, 256, k=3, s=2, p=1) # /16
        self.layer7 = C2f(256, 256, n=1, shortcut=True)
        self.layer8 = SPPF(256, 256)                    # P5 out (256 ch)

        # --- 2. Neck (PANet) ---
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Top-down path (FPN)
        # P5 (256) -> P4 (128)
        self.p5_conv = ConvBlock(256, 128, k=1, s=1, p=0) # Reduce P5 channels
        self.n1_c2f = C2f(128 + 128, 128, n=1) # Concat(up(P5), P4)
        
        # P4 (128) -> P3 (64)
        self.n1_conv = ConvBlock(128, 64, k=1, s=1, p=0) # Reduce N1 channels
        self.n2_c2f = C2f(64 + 64, 64, n=1)   # Concat(up(N1), P3) -> N2 (Neck Out 1)

        # Bottom-up path (PAN)
        # P3 (64) -> P4 (128)
        self.n2_down = ConvBlock(64, 64, k=3, s=2, p=1) # Downsample N2
        self.n3_c2f = C2f(64 + 128, 128, n=1) # Concat(down(N2), N1) -> N3 (Neck Out 2)
        
        # P4 (128) -> P5 (256)
        self.n3_down = ConvBlock(128, 128, k=3, s=2, p=1) # Downsample N3
        self.n4_c2f = C2f(128 + 128, 256, n=1) # Concat(down(N3), p5_conv) -> N4 (Neck Out 3)

        # --- 3. Head ---
        # Apply a SimpleDecoupledHead to each of the 3 neck outputs
        self.head1 = SimpleDecoupledHead(64, num_classes)  # For N2 (small objects)
        self.head2 = SimpleDecoupledHead(128, num_classes) # For N3 (medium objects)
        self.head3 = SimpleDecoupledHead(256, num_classes) # For N4 (large objects)

    def forward(self, x):
        # 1. Backbone
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        p3 = self.layer3(l2)  # P3 feature map ( /4 scale, 64 channels)
        
        l4 = self.layer4(p3)
        p4 = self.layer5(l4)  # P4 feature map ( /8 scale, 128 channels)
        
        l6 = self.layer6(p4)
        l7 = self.layer7(l6)
        p5 = self.layer8(l7)  # P5 feature map (/16 scale, 256 channels)
        
        # 2. Neck (PANet)
        # Top-down
        p5_up = self.p5_conv(p5)                    # (128 ch)
        n1 = self.n1_c2f(torch.cat([self.up(p5_up), p4], 1)) # (128 ch)
        
        n1_up = self.n1_conv(n1)                    # (64 ch)
        n2 = self.n2_c2f(torch.cat([self.up(n1_up), p3], 1)) # (64 ch) -> Neck Out 1
        
        # Bottom-up
        n3 = self.n3_c2f(torch.cat([self.n2_down(n2), n1], 1)) # (128 ch) -> Neck Out 2
        n4 = self.n4_c2f(torch.cat([self.n3_down(n3), p5_up], 1)) # (256 ch) -> Neck Out 3
        
        # 3. Head
        out1 = self.head1(n2) # Head for small objects (/4 scale)
        out2 = self.head2(n3) # Head for medium objects (/8 scale)
        out3 = self.head3(n4) # Head for large objects (/16 scale)
        
        # Return a list of the 3 head outputs
        return [out1, out2, out3]

# --- Example Usage ---
if __name__ == "__main__":
    num_classes = 80
    model = YOLOv8Scratch(num_classes=num_classes)
    
    # Create a dummy input tensor
    # Batch size = 2, Channels = 3, Height = 640, Width = 640
    dummy_input = torch.randn(2, 3, 640, 640)
    
    print(f"Input shape: {dummy_input.shape}")
    
    try:
        outputs = model(dummy_input)
        
        print(f"\nModel instantiated successfully.")
        print(f"Output is a list of {len(outputs)} tensors:")
        
        # Expected output shapes:
        # H, W = 640, 640
        # Channels = 4 (box) + 80 (classes) = 84
        
        # Head 1 (from N2, /4 scale)
        # 640/4 = 160
        shape1 = (2, 84, 160, 160)
        print(f"  Output 1 shape: {outputs[0].shape} (Expected: {shape1})")
        assert outputs[0].shape == shape1
        
        # Head 2 (from N3, /8 scale)
        # 640/8 = 80
        shape2 = (2, 84, 80, 80)
        print(f"  Output 2 shape: {outputs[1].shape} (Expected: {shape2})")
        assert outputs[1].shape == shape2

        # Head 3 (from N4, /16 scale)
        # 640/16 = 40
        shape3 = (2, 84, 40, 40)
        print(f"  Output 3 shape: {outputs[2].shape} (Expected: {shape3})")
        assert outputs[2].shape == shape3
        
        print("\nForward pass successful! Model is ready.")
        
    except Exception as e:
        print(f"\nError during forward pass: {e}")
import torch
import torch.nn as nn
from typing import List

class ConvBlock(nn.Module):
    """
    Standard Convolution Block: Conv2d + BatchNorm2d + SiLU Activation
    (Re-using your provided block)
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
    (Re-using your provided block)
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
    Modern C2f Block (replaces C3 from YOLOv5/your code)
    This is a key component of models like YOLOv8.
    'e' (expansion) controls the hidden dimension.
    """
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        
        # 1. Initial 1x1 conv to create the split point
        # Output channels = 2 * c (one for skip, one for bottleneck chain)
        self.cv1 = ConvBlock(c1, 2 * self.c, k=1, s=1, p=0)
        
        # 2. Final 1x1 conv to fuse all features
        # Input channels = 2*c (from initial split) + n*c (from n bottlenecks)
        self.cv2 = ConvBlock((2 + n) * self.c, c2, k=1, s=1, p=0)
        
        # 3. The 'n' bottleneck modules
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n))

    def forward(self, x):
        # 1. Apply initial conv
        y = self.cv1(x)
        
        # 2. Split the features into two parts
        # y1 is the skip connection, y2 goes through the bottlenecks
        y1, y2 = y.split((self.c, self.c), 1)
        
        # 3. Create a list of features to fuse, starting with the two splits
        fused_features: List[torch.Tensor] = [y1, y2]
        
        # 4. Pass y2 through all bottlenecks, appending each result
        for m in self.m:
            y2 = m(y2)
            fused_features.append(y2)
        
        # 5. Concatenate all features and apply final fusion conv
        return self.cv2(torch.cat(fused_features, 1))

class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF)
    (Re-using your provided block, with a slight correction:
     removed torch.no_grad() as pooling should be part of the
     gradient graph during training)
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
    A simplified, anchor-free, decoupled head (common in modern YOLOs).
    This replaces the simple coupled, anchor-based head from your YOLOv5 example.
    
    It outputs a single tensor of shape [B, 4 + num_classes, H, W].
    - 4 channels for BBox regression (e.g., l, t, r, b offsets)
    - num_classes channels for classification
    """
    def __init__(self, in_channels, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        c_ = 128 # Hidden channels, like in your v5 head
        
        # Common 1x1 conv, analogous to your v5 head's first layer
        self.common_conv = ConvBlock(in_channels, c_, k=1, s=1, p=0)
        
        # Regression branch
        # Outputs 4 channels for (left, top, right, bottom) or (x, y, w, h)
        self.reg_head = nn.Conv2d(c_, 4, 1)
        
        # Classification branch
        # Outputs num_classes channels
        self.cls_head = nn.Conv2d(c_, num_classes, 1)

    def forward(self, x):
        # Apply common layer
        x = self.common_conv(x)
        
        # Get decoupled outputs
        reg_output = self.reg_head(x)
        cls_output = self.cls_head(x)
        
        # Concatenate for a single output tensor
        # Shape: [Batch_Size, 4 + Num_Classes, H, W]
        return torch.cat([reg_output, cls_output], dim=1)

class YOLOModernScratch(nn.Module):
    """
    A simplified 'from scratch' model incorporating modern YOLO concepts
    (like C2f blocks and a decoupled head) as a "YOLOv12" example.
    
    This model is now anchor-free.
    """
    def __init__(self, num_classes=80):
        super().__init__()
        
        # --- Simple Backbone ---
        # We replace C3 with C2f
        self.layer1 = ConvBlock(3, 32, k=6, s=2, p=2)   # Stem
        self.layer2 = ConvBlock(32, 64, k=3, s=2, p=1)
        self.layer3 = C2f(64, 64, n=1, shortcut=True)   # REPLACED C3
        self.layer4 = ConvBlock(64, 128, k=3, s=2, p=1)
        self.layer5 = C2f(128, 128, n=2, shortcut=True) # REPLACED C3
        self.layer6 = ConvBlock(128, 256, k=3, s=2, p=1)
        self.layer7 = C2f(256, 256, n=1, shortcut=True) # REPLACED C3
        self.layer8 = SPPF(256, 256)                    # Kept SPPF

        # --- Modern Decoupled, Anchor-Free Head ---
        # This head outputs [B, 4 + num_classes, H, W]
        self.head = SimpleDecoupledHead(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.head(x)
        return x

# --- Example Usage ---
if __name__ == "__main__":
    num_classes = 80
    model = YOLOModernScratch(num_classes=num_classes)
    
    # Create a dummy input tensor
    # Batch size = 2, Channels = 3, Height = 640, Width = 640
    dummy_input = torch.randn(2, 3, 640, 640)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Perform a forward pass
    try:
        output = model(dummy_input)
        # The backbone downsamples by 2^5 = 32
        # 640 / 32 = 20
        # Expected output shape: [B, 4 + num_classes, H/32, W/32]
        # [2, 84, 20, 20]
        print(f"Output shape: {output.shape}")
        
        expected_channels = 4 + num_classes
        assert output.shape == (2, expected_channels, 20, 20)
        print("\nModel instantiated and forward pass successful!")
        
    except Exception as e:
        print(f"\nError during forward pass: {e}")

    # You can print the model summary
    # print(model)
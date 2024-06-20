import torch
import torch.nn as nn

from unet_parts import DoubleConv, DownBlock, UpBlock

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # Downsampling path
        self.down_block1 = DownBlock(in_channels, 64)
        self.down_block2 = DownBlock(64, 128)
        self.down_block3 = DownBlock(128, 256)
        self.down_block4 = DownBlock(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Upsampling path
        self.up_block1 = UpBlock(1024, 512)
        self.up_block2 = UpBlock(512, 256)
        self.up_block3 = UpBlock(256, 128)
        self.up_block4 = UpBlock(128, 64)

        # Output layer
        self.output_conv = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        # Downsampling path
        down1, pooled1 = self.down_block1(x)
        down2, pooled2 = self.down_block2(pooled1)
        down3, pooled3 = self.down_block3(pooled2)
        down4, pooled4 = self.down_block4(pooled3)
        
        # Bottleneck
        bottleneck = self.bottleneck(pooled4)

        # Upsampling path
        up1 = self.up_block1(bottleneck, down4)
        up2 = self.up_block2(up1, down3)
        up3 = self.up_block3(up2, down2)
        up4 = self.up_block4(up3, down1)  
        
        # Output layer
        output = self.output_conv(up4)
        return output
    
    
if __name__ == '__main__':
    double_conv = DoubleConv(256, 256)
    print(double_conv)
        
    input_image = torch.rand((1, 3, 512, 512))
    model = UNet(3, 10)
    output = model(input_image)
    print(output.size()) # Expected [1, 10, 512, 512]
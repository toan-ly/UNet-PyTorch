import torch 
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    '''
    Applies two convolution layers with ReLU activation.
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    '''
    Downsampling block with DoubleConv followed by MaxPooling.
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv_output = self.double_conv(x)
        pooled_output = self.max_pool(conv_output)

        return conv_output, pooled_output

class UpBlock(nn.Module):
    '''
    Upsampling block with ConvTranspose2d followed by DoubleConv.
    '''    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.double_conv(x)

    def _crop_and_concat(self, x1, x2):
        '''
        Crop x1 to the size of x2 and concatenate them along the channel dimension.
        '''
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x//2, diff_x - diff_x//2, 
                        diff_y//2, diff_y - diff_y//2])

        return torch.cat([x2, x1], dim=1)
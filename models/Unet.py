import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as T



# Double Conv
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False) # bias=False
        self.bn1 = nn.BatchNorm2d(out_channels)   
        self.relu1 = nn.ReLU(inplace=True)  # inplace=True
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) 
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # downs
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # ups
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)) # upsampling
            self.ups.append(DoubleConv(feature*2, feature)) # 因為對面會有相同channel的特徵來進行skip connection 所以輸入還是feature*2
        self.final_conv = nn.Conv2d(features[0], out_channels, 1) # 利用1*1 conv 改變channel數



    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, (2, 2))

        x = self.bottleneck(x)
        skip_connections.reverse()

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]
            x = torch.cat((skip_connection, x), dim=1) # B*C*H*W channel方向合併 dim=1
            x = self.ups[i + 1](x)
        
        out = self.final_conv(x)

        return out
# TEST    
# model = UNet()
# toy_data = torch.ones((16, 3, 224, 224))
# out = model(toy_data)
# print(out.shape)


  





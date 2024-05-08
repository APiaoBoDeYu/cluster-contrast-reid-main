import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

class BasicConv2d(nn.Module):
 
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
       
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionD(nn.Module):
 
    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, padding=1, stride=2)
 
        self.branch3x3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3x3_2 = conv_block(192, 192, kernel_size=3, padding=1,stride=2)
        self.branch3x3x3_3 = conv_block(192, 192, kernel_size=3, padding=1,stride=2)
        self.branch3x3x3_4 = conv_block(192, 192, kernel_size=3, stride=2)
 
    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
 
        branch3x3x3 = self.branch3x3x3_1(x)
        branch3x3x3 = self.branch3x3x3_3(branch3x3x3)
 
        branch_pool = F.max_pool2d(x, kernel_size=2, stride=2)
        outputs = [branch3x3, branch3x3x3, branch_pool]
        return outputs
 
    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)
    

# 示例使用
if __name__ == '__main__':
    inchinel=512
    wh=14
    block = InceptionD(inchinel)  # 实例化Coordinate Attention模块
    input = torch.rand(1, inchinel, wh, wh)  # 创建一个随机输入
    output = block(input)  # 通过模块处理输入
    print(output.shape)  # 打印输入和输出的尺寸
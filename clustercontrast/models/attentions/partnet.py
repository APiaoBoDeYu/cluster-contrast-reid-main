import numpy as np
import torch
from torch import nn
from torch.nn import init
from .sea import SEAttention
from .inception import InceptionD
from .scse import scSE
from .eca import ECAAttention

class ParNetAttention(nn.Module):
    # 初始化ParNet注意力模块
    def __init__(self, channel=256,nexchannel=768):
        super().__init__()
        # 使用自适应平均池化和1x1卷积实现空间压缩，然后通过Sigmoid激活函数产生权重图
        #定义Inception
        self.inception=InceptionD(channel)
        #定义scse
        self.scse=scSE(nexchannel)
        #定义eca

    def forward(self, x):
        ince_out=self.inception(x)
        scse=self.scse(ince_out)
        return scse


# 测试ParNetAttention模块
if __name__ == '__main__':
    input = torch.randn(3, 256, 7, 7)  # 创建一个随机输入  
    pna = ParNetAttention(channel=256,nexchannel=768)  # 实例化ParNet注意力模块
    output = pna(input)  # 对输入进行处理
    print(output.shape)  # 打印输出的形状，预期为(3, 512, 7, 7)
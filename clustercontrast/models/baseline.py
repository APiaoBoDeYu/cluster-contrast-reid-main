

import torch
from torch import nn


# from .resnet import ResNet, BasicBlock, Bottleneck
# from clustercontrast.models.attentions.myresnet import ResNet, BasicBlock, Bottleneck
from clustercontrast.models.attentions.myresnet import ResNet, BasicBlock, Bottleneck
from clustercontrast.models.attentions.sea import SEAttention
from clustercontrast.models.attentions.inception import InceptionD
from clustercontrast.models.attentions.scse import scSE
from clustercontrast.models.attentions.eca import ECAAttention
from clustercontrast.models.attentions.partnet import ParNetAttention
# from .attentions.myresnet import ResNet, BasicBlock, Bottleneck
# from .attentions.sea import SEAttention
# from .attentions.inception import InceptionD
# from .attentions.scse import scSE
# from .attentions.eca import ECAAttention
# from .attentions.partnet import ParNetAttention
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()

        self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
 
        
        #获取resnet50的所有层
        resnet50_layers=list(self.base.children())
        #选择截取的位置
        #位置3
        self.reslayer3=nn.Sequential(*resnet50_layers[:4])
        #位置4
        self.reslayer4=resnet50_layers[4]
        #位置5
        self.reslayer5=resnet50_layers[5]
        #位置6
        self.reslayer6=resnet50_layers[6]

        #定义基本算子
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv1=nn.Conv2d(3840, 2048, kernel_size=1)


        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.num_features = 2048
        self.neck = neck
        self.neck_feat = neck_feat

        #定义分支1
        self.parnet1=ParNetAttention(channel=256,nexchannel=768) 
        #定义分支2
        self.parnet2=ParNetAttention(channel=512,nexchannel=1024) 

        


        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        #获取中间结果
        reslayer3_out=self.reslayer3(x)
        reslayer4_out=self.reslayer4(reslayer3_out)
        reslayer5_out=self.reslayer5(reslayer4_out)
        reslayer6_out=self.reslayer6(reslayer5_out)

        #计算第一个分支
        branch1=self.parnet1(reslayer3_out)
        branch2=self.parnet2(reslayer4_out)
        branch1=self.maxpool(branch1)

        #将所有分支拼接在一起
        all_branch=torch.cat([branch1,branch2,reslayer6_out], 1)
        #将维度提取到2048
        connected=self.conv1(all_branch)

        global_feat = self.gap(connected)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        return global_feat

        # if self.neck == 'no':
        #     feat = global_feat
        # elif self.neck == 'bnneck':
        #     feat = self.bottleneck(global_feat)  # normalize for angular softmax

        # if self.training:
        #     cls_score = self.classifier(feat)
        #     return cls_score, global_feat  # global feature for triplet loss
        # else:
        #     if self.neck_feat == 'after':
        #         # print("Test with feature after BN")
        #         return feat
        #     else:
        #         # print("Test with feature before BN")
        #         return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

if __name__ == '__main__':
    model = Baseline(751, 1, '/root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth', 'bnneck', 'after', 'resnet50', 'imagenet')
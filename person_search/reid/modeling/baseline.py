# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from IPython import embed
import torch
import torchvision
from IPython import embed
from torch import nn
import sys
sys.path.append("/ReID")
# print(sys.path)
from modeling.backbones.resnet import BasicBlock, Bottleneck,ResNet
from modeling.backbones.resnet_ibn_a import resnet50_ibn_a
from modeling.backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, model_name, model_path, last_stride=1):
        """
        :param num_classes: 10126 训练的行人ID类别数目
        :param model_name: 'resnet50_ibn_a'
        :param model_path: 预训练模型路径 '/home/common/wangsong/weights/r50_ibn_a.pth'
        :param last_stride: 1 取消最后的下采样
        :param neck: 使用'bnneck'
        :param neck_feat: 'after'
        """

        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)

        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        self.gap = nn.AdaptiveAvgPool2d(1) #全局平局池化层，输出特征图进行空间池化，将其转换为固定大小的表示（1x1 空间尺寸）
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes

        self.bottleneck = nn.BatchNorm1d(self.in_planes) #对指定特征维度2048进行归一化，全局特征进行归一化操作，增强模型的性能和鲁棒性，并促进模型的收敛和泛化能力
        self.bottleneck.bias.requires_grad_(False)  # no shift 避免训练过程中不必要的更新
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False) #分类器,全连接层，将特征映射到类别数目

    def forward(self, x):
        x = self.base(x) #主干网络
        global_feat = self.gap(x)  # (b, 2048, 1, 1)，将多维的特征图转换为固定长度的全局特征向量,全局信息捕捉并提高模型泛化能力
        # flatten to (bs, 2048)，将全局特征向量的形状从多维张量转换为二维张量，以适应后续操作或模块的输入要求
        global_feat = global_feat.view(global_feat.shape[0], -1)

        feat = self.bottleneck(global_feat)  # normalize for angular softmax

        # print("Test with feature after BN")
        return feat


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path,map_location=torch.device('cpu')).state_dict()
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])




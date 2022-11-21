import torch
from model.backbone import resnet
import numpy as np
class conv_bn_relu(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size, 
            stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class parsingNet(torch.nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='50', use_aux=False,use_sad=False):
        super(parsingNet, self).__init__()

        self.size = size
        self.w = size[0]
        self.h = size[1]
#        self.cls_dim = cls_dim # (num_gridding, num_cls_per_lane, num_of_lanes)
        # num_cls_per_lane is the number of row anchors
        self.use_aux = use_aux
        self.use_sad = use_sad
#        self.total_dim = np.prod(cls_dim)

        # input : nchw,
        # output: (w+1) * sample_rows * 4 
        self.model = resnet(backbone, pretrained=pretrained)
#        self.model = EfficientNet.from_pretrained(backbone, num_classes=24)
        if self.use_aux:
            self.aux_header2 = torch.nn.Sequential(
                conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_header3 = torch.nn.Sequential(
                conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                conv_bn_relu(384, 256, 3,padding=2,dilation=2),
                conv_bn_relu(256, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=4,dilation=4),
                torch.nn.Conv2d(128, 5,1)
                # output : n, num_of_lanes+1, h, w
            )
            initialize_weights(self.aux_header2,self.aux_header3,self.aux_header4,self.aux_combine)

        self.soft = torch.nn.Softmax(dim=1)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(225,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 5)
        )
        self.cls1 = torch.nn.Sequential(
            conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1)
        )
        self.cls2 = torch.nn.Sequential(
            conv_bn_relu(128, 4, kernel_size=3, stride=1, padding=1)
        )

        self.eff = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            torch.nn.Dropout(p=0.2,inplace=False),
            torch.nn.Sigmoid()
        )
        self.eff2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=4*225,out_features=256,bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256,out_features=28,bias=True) 
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()


    def forward(self, x):
        # n c h w - > n 2048 sh sw
        # -> n 2048
        x2,x3,fea = self.model(x)
        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3_1 = torch.nn.functional.interpolate(x3,scale_factor = 2,mode='bilinear')
            x4 = self.aux_header4(fea)
            x4_1 = torch.nn.functional.interpolate(x4,scale_factor = 4,mode='bilinear')
            aux_seg = torch.cat([x2,x3_1,x4_1],dim=1)
            aux_seg = self.aux_combine(aux_seg)
        else:
            aux_seg = None

#poly strat
        lane_all = self.cls1(fea)
        lane_all2 = self.cls2(lane_all)
        lane_all3 = self.eff2(lane_all2.view(-1,4*225))

        lane_all3 = lane_all3.view(-1,4,7)
        lane = lane_all3[:,:,1:5]

        bin_cls =self.sigmoid(lane_all3[:,:,0]).reshape(-1,4)
        start = self.sigmoid(lane_all3[:,:,5]).reshape(-1,4)
        end = self.sigmoid(lane_all3[:,:,6]).reshape(-1,4)
#poly end


        if self.use_aux:
            return lane, bin_cls, aux_seg,x2,x3,x4,start,end,lane_all,lane_all2
          #      return aux_seg,x2,x3,f
        else:
            return lane, bin_cls,start,end

        return lane,bin_cls


def initialize_weights(*models):
    for model in models:
        real_init_weights(model)
def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):    
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)

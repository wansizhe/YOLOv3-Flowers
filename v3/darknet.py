import torch
from torch import nn
import math
from collections import OrderedDict

class BasicBlock(nn.Module):
    '''
    res unit = input + (DBL * 2)
    '''
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        '''
        DBL = conv + bn + leakyrelu
        '''
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False) # 下降通道数
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)    # 恢复通道数
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x
        # DBL 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        # DBL 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out

class DarkNet(nn.Module):
    '''
    DarkNet-53 without FC layer
    '''
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        '''
        DBL = conv + bn + leakyrelu
        '''
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        '''
        res1, res2, res8, res8, res4
        '''
        self.layer1 = self._make_layer([32, 64], layers[0])
        self.layer2 = self._make_layer([64, 128], layers[1])
        self.layer3 = self._make_layer([128, 256], layers[2])
        self.layer4 = self._make_layer([256, 512], layers[3])
        self.layer5 = self._make_layer([512, 1024], layers[5])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks):
        '''
        resn = zero padding + DBL + (res * n)
        '''
        layers = []
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))

        self.inplanes = planes[1]
        for i in range(blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        '''
        | (416)DBL
        | (208)res1
        | (104)res2
        |  (52)res8 ——> out3
        |  (26)res8 ——> out4
        |  (13)res4 ——> out5
        '''
        # DBL
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # res1
        x = self.layer2(x)  # res2
        out3 = self.layer3(x)   # res8
        out4 = self.layer4(out3)    # res8
        out5 = self.layer5(out4)    # res4

        return out3, out4, out5

def darknet53(pretrained=False, **kwargs):
    model = DarkNet([1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception('darknet request a pretrained path. got [{}]'.format(pretrained))
    return model

import torch.nn as nn
import math

__all__ = ['MobileNetV2', 'mobilenetv2_19']

class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, expansion=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes*expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes*expansion)
        self.conv2 = nn.Conv2d(inplanes*expansion, inplanes*expansion, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=inplanes*expansion)
        self.bn2 = nn.BatchNorm2d(inplanes*expansion)
        self.conv3 = nn.Conv2d(inplanes*expansion, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MobileNetV2(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 32
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1, expansion = 1)
        self.layer2 = self._make_layer(block, 24, layers[1], stride=2, expansion = 6)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2, expansion = 6)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2, expansion = 6)
        self.layer5 = self._make_layer(block, 96, layers[4], stride=1, expansion = 6)
        self.layer6 = self._make_layer(block, 160, layers[5], stride=2, expansion = 6)
        self.layer7 = self._make_layer(block, 320, layers[6], stride=1, expansion = 6)
        self.conv8 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, bias=False)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.conv9 = nn.Conv2d(1280,num_classes, kernel_size=1, stride=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, expansion):

        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, expansion=expansion))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.conv8(x)
        x = self.avgpool(x)
        x = self.conv9(x)
        x = x.view(x.size(0),-1)

        return x


def mobilenetv2_19(**kwargs):
    """Constructs a MobileNetV2-19 model.
    """
    model = MobileNetV2(Bottleneck, [1, 2, 3, 4, 3, 3, 1], **kwargs)
    return model

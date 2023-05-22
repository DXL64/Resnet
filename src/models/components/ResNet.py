import torch
import torch.nn as nn
import torch.amp
import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms
import math
from tqdm import tqdm
import torch.optim as optim
import csv
import os

def conv1x1(in_channels, out_channels, stride, padding, _bias = True):
  return nn.Conv2d(in_channels, out_channels, 1, stride, padding, bias = _bias)
def conv3x3(in_channels, out_channels, stride, padding, _bias = True):
  return nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias = _bias)

class BasicResBlock(nn.Module):
  expansion = 1
  def __init__(self, in_channels, out_channels, down_sample = None, stride = 1):
    super().__init__()
    self.conv1 = conv3x3(in_channels, out_channels, stride = stride, padding = 1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = conv3x3(out_channels, out_channels, stride = 1, padding = 1)
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.down_sample = down_sample
    self.stride = stride
    self.relu = nn.ReLU(inplace=True)
  def forward(self, x):
    identity = x
    # print(x.shape, "identity-------------------------")    
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)                                                                                                 
    out = self.bn2(out) 
    # print(out.shape, "out-------------------------")                                                 

    #note
    if self.down_sample is not None:
      identity = self.down_sample(x)

    # try:
      out += identity
    # except:
    #   import pdb; pdb.set_trace()
    #   print("v")
    out = self.relu(out)

    return out
    #note

class BottleNeckResBlock(nn.Module):
  expansion = 4
  def __init__(self, in_channels, out_channels, down_sample = None, stride = 1):
    super().__init__()
    # padding dieu chinh phu hop voi kernel_size
    self.conv1 = conv1x1(in_channels, out_channels, stride=1, padding=0)
    self.bn1 = nn.BatchNorm2d(out_channels)
        
    self.conv2 = conv3x3(out_channels, out_channels, stride=stride, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
        
    self.conv3 = conv1x1(out_channels, out_channels*self.expansion, stride=1, padding=0)
    self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        
    self.down_sample = down_sample
    self.stride = stride
    self.relu = nn.ReLU(inplace=True)
  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.down_sample is not None:
        identity = self.down_sample(identity)

    out += identity
    out = self.relu(out)

    return out
  
class ResNet(nn.Module):
  def __init__(self, ResResBlock, layer_list, num_classes, num_channels = 1):
    super().__init__()
    self.in_channels = 64
    
    self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.batch_norm1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
    self.layer1 = self._make_layer(ResResBlock, layer_list[0], planes=64)
    self.layer2 = self._make_layer(ResResBlock, layer_list[1], planes=128, stride=2)
    self.layer3 = self._make_layer(ResResBlock, layer_list[2], planes=256, stride=2)
    self.layer4 = self._make_layer(ResResBlock, layer_list[3], planes=512, stride=2)
        
    # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.avgpool = nn.AvgPool2d(7, stride=1)
    self.fc = nn.Linear(512*ResResBlock.expansion, num_classes)

    for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()    

  def forward(self, x):
    x = self.conv1(x)
    x = self.batch_norm1(x)
    x = self.relu(x)

    x = self.max_pool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
        
    # x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    # x = torch.nn.functional.softmax(x, dim=1)
    return x
            
  def _make_layer(self, ResBlock, blocks, planes, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers = []
        layers.append(ResBlock(self.in_channels, planes, down_sample=downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion
        for i in range(1, blocks):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)
def ResNet18(num_classes, channels):  
    return ResNet(BasicResBlock, [2,2,2,2], num_classes, num_channels=channels)

def ResNet34(num_classes, channels):
    return ResNet(BasicResBlock, [3,4,6,3], num_classes, num_channels=channels)

def ResNet50(num_classes, channels):
    return ResNet(BottleNeckResBlock, [3,4,6,3], num_classes, channels)
    
def ResNet101(num_classes, channels):
    return ResNet(BottleNeckResBlock, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels):
    return ResNet(BottleNeckResBlock, [3,8,36,3], num_classes, channels)

class ResnetModel(nn.Module):
    def __init__(
        self,
        num_classes = 10,
        channels = 1,
        resnet_version = 50
    ):
        super().__init__()
        print("--------------ResNet Version:-------------", resnet_version)
        # os.environ["CUDA_VISIBLE_DEVICES"] = "2"

        if resnet_version == 18:
          self.model = ResNet18(num_classes=num_classes, channels=channels)
        if resnet_version == 34:
          self.model = ResNet34(num_classes=num_classes, channels=channels)
        if resnet_version == 50:
          self.model = ResNet50(num_classes=num_classes, channels=channels)
        if resnet_version == 101:
          self.model = ResNet101(num_classes=num_classes, channels=channels)
        if resnet_version == 152: 
          self.model = ResNet152(num_classes=num_classes, channels=channels)

    def forward(self, x):
      # print(x.shape, "forward.................................")
      return self.model(x)

if __name__ == "__main__":
    net = ResnetModel(num_classes=10, channels=1)

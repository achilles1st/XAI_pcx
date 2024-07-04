import torch.nn as nn

class Model(nn.Module):
  def __init__(self, with_softmax=False):
    super().__init__()

    self.with_softmax = with_softmax

    self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
    self.conv1_relu = nn.ReLU()
    self.down_conv_pad1 = nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0.)
    self.down_conv1 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=2)
    self.down_conv1_relu = nn.ReLU()

    self.conv2 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
    self.conv2_relu = nn.ReLU()
    self.down_conv_pad2 = nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0.)
    self.down_conv2 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=2)
    self.down_conv2_relu = nn.ReLU()

    self.conv3 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
    self.conv3_relu = nn.ReLU()
    self.down_conv_pad3 = nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0.)
    self.down_conv3 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=2)
    self.down_conv3_relu = nn.ReLU()

    self.conv4 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
    self.conv4_relu = nn.ReLU()
    self.down_conv_pad4 = nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0.)
    self.down_conv4 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=2)
    self.down_conv4_relu = nn.ReLU()

    self.conv5 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
    self.conv5_relu = nn.ReLU()
    self.down_conv_pad5 = nn.ConstantPad3d((0, 1, 0, 1, 1, 1), 0.)
    self.down_conv5 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=2)
    self.down_conv5_relu = nn.ReLU()

    self.flatten = nn.Flatten()

    self.dens1 = nn.Linear(in_features=896, out_features=16)
    self.dens1_relu = nn.ReLU()

    self.dens2 = nn.Linear(in_features=16, out_features=2)
    self.dens2_softmax = nn.Softmax(dim=1)
  
  def forward(self, x):
    e1 = self.conv1_relu(self.conv1(x))
    de1 = self.down_conv1_relu(self.down_conv1(self.down_conv_pad1(e1)))
    
    e2 = self.conv2_relu(self.conv2(de1))
    de2 = self.down_conv2_relu(self.down_conv2(self.down_conv_pad2(e2)))
    
    e3 = self.conv3_relu(self.conv3(de2))
    de3 = self.down_conv3_relu(self.down_conv3(self.down_conv_pad3(e3)))
    
    e4 = self.conv4_relu(self.conv4(de3))
    de4 = self.down_conv4_relu(self.down_conv4(self.down_conv_pad4(e4)))
    
    e5 = self.conv5_relu(self.conv5(de4))
    de5 = self.down_conv5_relu(self.down_conv5(self.down_conv_pad5(e5)))
    
    f = self.flatten(de5.permute((0, 2, 3, 4, 1)))

    d1 = self.dens1_relu(self.dens1(f))
    logits = self.dens2(d1)

    if self.with_softmax:
      return self.dens2_softmax(logits)
    return logits
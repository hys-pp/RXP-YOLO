import torch
import torch.nn as nn

_all_ = ['XADNet']


class XADNet(nn.Module):
    def __init__(self, channels, num_of_layers):
        super(XADNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = num_of_layers
        groups = 1

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            self._make_layers(features, kernel_size, num_of_layers),
            nn.Conv2d(in_channels=features, out_channels=3, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False)
        )

        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.Tanh = nn.Tanh()

        # Initialize weights
        self._initialize_weights()

    def _make_layers(self, features, kernel_size, num_of_layers):
        layers = []
        for _ in range(num_of_layers):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x1 = self.conv_layers(x)
        out = torch.cat([input, x1], 1)
        out = self.Tanh(out)
        out = self.conv3(out)
        out = out * x1
        out2 = input - out
        return out2

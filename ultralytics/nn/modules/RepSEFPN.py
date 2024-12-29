import torch
import torch.nn as nn
import numpy as np


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = swish()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class RepConv(nn.Module):
    default_act = swish()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) block."""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.fc1(x_se)
        x_se = self.fc2(x_se)
        x_se = self.sigmoid(x_se)
        return x * x_se


class CSPEStage(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 n,
                 block_fn='BasicBlock_3x3_Reverse',
                 ch_hidden_ratio=1.0,
                 act='silu',
                 spp=False,
                 se_block=False):
        super(CSPEStage, self).__init__()

        split_ratio = 2
        ch_first = int(ch_out // split_ratio)
        ch_mid = int(ch_out - ch_first)
        self.conv1 = Conv(ch_in, ch_first, 1)
        self.conv2 = Conv(ch_in, ch_mid, 1)
        self.convs = nn.Sequential()

        self.se_block = SEBlock(ch_mid) if se_block else None

        next_ch_in = ch_mid
        for i in range(n):
            self.convs.add_module(
                str(i),
                BasicBlock_3x3_Reverse(next_ch_in,
                                       ch_hidden_ratio,
                                       ch_mid,
                                       shortcut=True))
            if i == (n - 1) // 2 and spp:
                self.convs.add_module('spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13]))
            next_ch_in = ch_mid
        self.conv3 = Conv(ch_mid * n + ch_first, ch_out, 1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)

        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)

        if self.se_block:
            y = self.se_block(y)

        y = self.conv3(y)
        return y


class RepSEFPN(nn.Module):
    def __init__(self, ch_in, ch_out, n=3, block_fn='BasicBlock_3x3_Reverse', act='silu', spp=False, se_block=False):
        super(RepSEFPN, self).__init__()

        self.stage1 = CSPEStage(ch_in, ch_out, n, block_fn, ch_hidden_ratio=1.0, act=act, spp=spp, se_block=se_block)
        self.stage2 = CSPEStage(ch_out, ch_out * 2, n, block_fn, ch_hidden_ratio=1.0, act=act, spp=spp, se_block=se_block)
        self.stage3 = CSPEStage(ch_out * 2, ch_out * 4, n, block_fn, ch_hidden_ratio=1.0, act=act, spp=spp, se_block=se_block)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x

class BasicBlock_3x3_Reverse(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_hidden_ratio,
                 ch_out,
                 shortcut=True):
        super(BasicBlock_3x3_Reverse, self).__init__()
        assert ch_in == ch_out
        ch_hidden = int(ch_in * ch_hidden_ratio)
        self.conv1 = Conv(ch_hidden, ch_out, 3, s=1)
        self.conv2 = RepConv(ch_in, ch_hidden, 3, s=1)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        if self.shortcut:
            return x + y
        else:
            return y


class SPP(nn.Module):
    def __init__(
            self,
            ch_in,
            ch_out,
            k,
            pool_size
    ):
        super(SPP, self).__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(kernel_size=size,
                                stride=1,
                                padding=size // 2,
                                ceil_mode=False)
            self.add_module('pool{}'.format(i), pool)
            self.pool.append(pool)
        self.conv = Conv(ch_in, ch_out, k)

    def forward(self, x):
        outs = [x]

        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)

        y = self.conv(y)
        return y

# Example usage:
model = RepSEFPN(3, 64, n=3, block_fn='BasicBlock_3x3_Reverse', act='silu', spp=True, se_block=True)



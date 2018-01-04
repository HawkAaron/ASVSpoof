from __future__ import print_function

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

"""Light CNN with res-block"""

class mfm(chainer.Chain):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        with self.init_scope():
            self.out_channels = out_channels
            if type == 1:
                self.filter = L.Convolution2D(in_channels, 2*out_channels, ksize=kernel_size,
                                              stride=stride, pad=padding)
            else:
                self.filter = L.Linear(in_channels, 2*out_channels)

    def __call__(self, x):
        x = self.filter(x)
        out_0, out_1 = F.split_axis(x, 2, 1)
        return F.maximum(out_0, out_1)


class group(chainer.Chain):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        with self.init_scope():
            self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
            self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def __call__(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x

class resblock(chainer.Chain):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        with self.init_scope():
            self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out


class LightCNN(chainer.Chain):
    '''
    light CNN 29 layers
    '''
    def __init__(self, num_classes=2):
        super(LightCNN, self).__init__()
        with self.init_scope():

            self.conv1 = mfm(1, 48, 5, 1, 2)
            #self.pool1 = F.MaxPooling2D(kernel_size=2, stride=2, ceil_mode=True)
            self.block1_1 = resblock(48, 48)
            self.group1 = group(48, 96, 3, 1, 1)
            #self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.block2_1 = resblock(96, 96)
            self.block2_2 = resblock(96, 96)
            self.group2 = group(96, 192, 3, 1, 1)
            #self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.block3_1 = resblock(192, 192)
            self.block3_2 = resblock(192, 192)
            self.block3_3 = resblock(192, 192)
            self.group3 = group(192, 128, 3, 1, 1)
            self.block4_1 = resblock(128, 128)
            self.block4_2 = resblock(128, 128)
            self.block4_3 = resblock(128, 128)
            self.block4_4 = resblock(128, 128)
            self.group4 = group(128, 128, 3, 1, 1)
            #self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            #change the input dim of mfm
            self.fc = mfm(25 * 25 * 128, 256, type=0)
            self.fc2 = L.Linear(256, num_classes)

    def __call__(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], -1)

        x = self.conv1(x)
        x = F.max_pooling_2d(x, ksize=2, stride=2)

        x = self.block1_1(x)
        x = self.group1(x)
        x = F.max_pooling_2d(x, ksize=2, stride=2)

        x = self.block2_2(self.block2_1(x))
        x = self.group2(x)
        x = F.max_pooling_2d(x, ksize=2, stride=2)

        x = self.block3_3(self.block3_2(self.block3_1(x)))
        x = self.group3(x)
        x = self.block4_4(self.block4_3(self.block4_2(self.block4_1(x))))
        x = self.group4(x)
        x = F.max_pooling_2d(x, ksize=2, stride=2)

        #x = x.view(x.size(0), -1)

        x.reshape((x.shape[0], -1))
        #print (x.shape)
        #
        fc = self.fc(x)
        fc = F.dropout(fc)
        out = self.fc2(fc)
        return out

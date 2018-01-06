from .common import *

class CONV_BLOCK(chainer.Chain):
    def __init__(self, out_channels=64, ksize=3, pad=1):
        super(CONV_BLOCK, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channels, ksize, pad=pad, nobias=True)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        h = F.leaky_relu(h)
        return F.dropout(h)

# Network definition
class CNN(chainer.Chain):

    def __init__(self, n_units=32, n_out=2):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = CONV_BLOCK(n_units, 3)
            self.conv2 = CONV_BLOCK(n_units, 3)
            self.conv3 = CONV_BLOCK(n_units, 3)
            self.conv4 = CONV_BLOCK(n_units, 3)
            self.conv5 = CONV_BLOCK(n_units, 3)
            self.conv6 = CONV_BLOCK(n_units, 3)
            self.conv7 = CONV_BLOCK(n_units, 3)

            self.fc1 = L.Linear(None, n_units, nobias=True)
            self.bn = L.BatchNormalization(n_units)
            self.fc2 = L.Linear(None, n_out, nobias=True)

    def __call__(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], -1)

        h = self.conv1(x)
        h = self.conv2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = self.conv3(h)
        h = self.conv4(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # h = self.conv5(h)
        # h = self.conv6(h)
        # h = self.conv7(h)
        # h = F.max_pooling_2d(h, ksize=2, stride=2)

        h.reshape(h.shape[0], -1)

        h = self.fc1(h)
        h = self.bn(h)
        h = F.leaky_relu(h)
        h = F.dropout(h)
        return self.fc2(h)
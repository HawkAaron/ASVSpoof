from .common import *

class CONV_BLOCK(chainer.Chain):
    def __init__(self, out_channels=64, ksize=11, pad=1):
        super(CONV_BLOCK, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channels, ksize, pad=pad,
                                        nobias=True)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return F.relu(h)

class DNN_BLOCK(chainer.Chain):
    def __init__(self, h_dim):
        super(DNN_BLOCK, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, h_dim)
            self.bn = L.BatchNormalization(h_dim)
    
    def __call__(self, x):
        h = self.l1(x)
        h = self.bn(h)
        h = F.relu(h)
        return F.dropout(h)

# Network definition
class CLD(chainer.Chain):

    def __init__(self, n_units=512, n_out=2):
        super(CLD, self).__init__()
        with self.init_scope():
            self.conv1 = CONV_BLOCK(4, 3)
            self.conv2 = CONV_BLOCK(4, 3)
            self.dnn1 = DNN_BLOCK(n_units)
            self.dnn2 = DNN_BLOCK(n_units)
            self.dnn3 = DNN_BLOCK(n_units)
            self.dnn4 = DNN_BLOCK(n_units)
            self.dnn5 = DNN_BLOCK(n_units)
            self.lout = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], -1)
        
        h = self.conv1(x)
        h = F.dropout(h)
        h = self.conv2(x)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.dropout(h)
        h = self.dnn1(h)
        h = self.dnn2(h)
        h = self.dnn3(h)
        h = self.dnn4(h)
        h = self.dnn5(h)
        return self.lout(h)
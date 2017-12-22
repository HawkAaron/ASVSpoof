from __future__ import print_function

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

'''
DNN
'''
class DNN(chainer.Chain):
    def __init__(self, h_dim):
        super(DNN, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, h_dim)
            self.bn1 = L.BatchNormalization(h_dim)
            
            self.l2 = L.Linear(None, h_dim)
            self.bn2 = L.BatchNormalization(h_dim)

            self.l3 = L.Linear(None, h_dim)
            self.bn3 = L.BatchNormalization(h_dim)

            self.l4 = L.Linear(None, 2)

    def __call__(self, x):
        h = self.l1(x)
        # h = self.bn1(h)
        h = F.relu(h)
        # h = F.dropout(h)

        h = self.l2(x)
        # h = self.bn2(h)
        h = F.relu(h)
        # h = F.dropout(h)

        h = self.l3(x)
        # h = self.bn3(h)
        h = F.relu(h)
        # h = F.dropout(h)
        return self.l4(h)

'''
CONV + DNN
'''
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

class MLP_BLOCK(chainer.Chain):
    def __init__(self, h_dim):
        super(MLP_BLOCK, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, h_dim)
            self.bn = L.BatchNormalization(h_dim)
    
    def __call__(self, x):
        h = self.l1(x)
        h = self.bn(h)
        h = F.relu(h)
        return F.dropout(h)

# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.conv1 = CONV_BLOCK(16, 3)
            self.conv2 = CONV_BLOCK(16, 3)
            self.b1 = MLP_BLOCK(n_units)
            self.b2 = MLP_BLOCK(n_units)
            self.b3 = MLP_BLOCK(n_units)
            self.b4 = MLP_BLOCK(n_units)
            self.b5 = MLP_BLOCK(n_units)
            self.lout = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        x = x.reshape(x.shape[0], 1, -1, 11)

        h = self.conv1(x)
        h = F.dropout(h)
        h = self.conv2(x)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.dropout(h)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        h = self.b4(h)
        h = self.b5(h)
        return self.lout(h)

'''
VGG
'''
class Block(chainer.Chain):

    """A convolution, batch norm, ReLU block.

    A block in a feedforward network that performs a
    convolution followed by batch normalization followed
    by a ReLU activation.

    For the convolution operation, a square filter size is used.

    Args:
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter is ksize x ksize.
        pad (int): The padding to use for the convolution.

    """

    def __init__(self, out_channels, ksize, pad=1):
        super(Block, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channels, ksize, pad=pad,
                                        nobias=True)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return F.relu(h)


class VGG(chainer.Chain):

    """A VGG-style network for very small images.

    This model is based on the VGG-style model from
    http://torch.ch/blog/2015/07/30/cifar.html
    which is based on the network architecture from the paper:
    https://arxiv.org/pdf/1409.1556v6.pdf

    This model is intended to be used with either RGB or greyscale input
    images that are of size 32x32 pixels, such as those in the CIFAR10
    and CIFAR100 datasets.

    On CIFAR10, it achieves approximately 89% accuracy on the test set with
    no data augmentation.

    On CIFAR100, it achieves approximately 63% accuracy on the test set with
    no data augmentation.

    Args:
        class_labels (int): The number of class labels.

    """

    def __init__(self, class_labels=10):
        super(VGG, self).__init__()
        with self.init_scope():
            self.block1_1 = Block(64, 3)
            self.block1_2 = Block(64, 3)
            self.block2_1 = Block(128, 3)
            self.block2_2 = Block(128, 3)
            # self.block3_1 = Block(256, 3)
            # self.block3_2 = Block(256, 3)
            # self.block3_3 = Block(256, 3)
            # self.block4_1 = Block(512, 3)
            # self.block4_2 = Block(512, 3)
            # self.block4_3 = Block(512, 3)
            # self.block5_1 = Block(512, 3)
            # self.block5_2 = Block(512, 3)
            # self.block5_3 = Block(512, 3)
            self.fc1 = L.Linear(None, 128, nobias=True)
            self.bn_fc1 = L.BatchNormalization(128)
            self.fc2 = L.Linear(None, class_labels, nobias=True)

    def __call__(self, x):
        x = x.reshape(x.shape[0], 1, -1, 11)
        
        # 64 channel blocks:
        h = self.block1_1(x)
        h = F.dropout(h, ratio=0.3)
        h = self.block1_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 128 channel blocks:
        h = self.block2_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block2_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # # 256 channel blocks:
        # h = self.block3_1(h)
        # h = F.dropout(h, ratio=0.4)
        # h = self.block3_2(h)
        # h = F.dropout(h, ratio=0.4)
        # h = self.block3_3(h)
        # h = F.max_pooling_2d(h, ksize=2, stride=2)

        # # 512 channel blocks:
        # h = self.block4_1(h)
        # h = F.dropout(h, ratio=0.4)
        # h = self.block4_2(h)
        # h = F.dropout(h, ratio=0.4)
        # h = self.block4_3(h)
        # h = F.max_pooling_2d(h, ksize=2, stride=2)

        # # 512 channel blocks:
        # h = self.block5_1(h)
        # h = F.dropout(h, ratio=0.4)
        # h = self.block5_2(h)
        # h = F.dropout(h, ratio=0.4)
        # h = self.block5_3(h)
        # h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.dropout(h, ratio=0.5)
        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        return self.fc2(h)

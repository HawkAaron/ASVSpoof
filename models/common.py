from __future__ import print_function
import chainer
from chainer import cuda
from chainer.dataset import concat_examples
from chainer.datasets import tuple_dataset
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import training
from chainer.training import extensions
import numpy as np
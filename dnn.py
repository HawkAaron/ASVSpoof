#!/usr/bin/env python
"""Fully-connected neural network example using MNIST dataset

This code is a custom loop version of train_mnist.py. That is, we train
models without using the Trainer class in chainer and instead write a
training loop that manually computes the loss of minibatches and
applies an optimizer to update the model.
"""
from __future__ import print_function

import argparse, os

import chainer
from chainer import cuda
from chainer.dataset import concat_examples
from chainer.datasets import tuple_dataset
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import training
from chainer.training import extensions
# from cqcc import load_cqcc
import numpy as np
from data_loader import DataSet, load_data, DataSetOnLine
from model import *

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--lr', '-l', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='dnn',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1024,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()

    try:
        os.mkdir(args.out)
    except:
        print('')

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    model = L.Classifier(MLP()) #MLP(args.unit, 2))
    if args.gpu >= 0:
        # Make a speciied GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))


    # extract all feature
    # train_data, train_label, _ = load_data()
    # train_data = np.vstack(train_data)
    # mean = np.mean(train_data, axis=0)
    # std = np.std(train_data, axis=0)
    # train_data = (train_data - mean) / std
    # dev_data, dev_label, _ = load_data(mode='dev')
    # dev_data = np.vstack(dev_data)
    # mean = np.mean(dev_data, axis=0)
    # std = np.std(dev_data, axis=0)
    # dev_data = (dev_data - mean) / std

    # train = DataSet(train_data, np.hstack(train_label))
    # dev = DataSet(np.vstack(dev_data), np.hstack(dev_label))

    train = DataSetOnLine(mode='train', feat_type='fft', buf=False)
    dev = DataSetOnLine(mode='dev', feat_type='fft', buf=False)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    dev_iter = chainer.iterators.SerialIterator(dev, args.batchsize, repeat=False, shuffle=False)

    sum_accuracy = 0
    sum_loss = 0
    train_count = 0
    dev_count = 0

    train_acc = []
    dev_acc = []


    while train_iter.epoch < args.epoch:
        batch = train_iter.next()   # feature: (batchsize, frames, window*dim)
        # Reduce learning rate by 0.5 every 25 epochs.
        # if train_iter.epoch % 5 == 0 and train_iter.is_new_epoch:
        #     optimizer.lr *= 0.5
        #     print('Reducing learning rate to: ', optimizer.lr)

        # x, t = concat_examples(batch)
        batch = np.array(batch)
        x = np.vstack(batch[:,0])
        t = np.hstack(batch[:,1])

        if args.gpu >= 0:
            x = cuda.to_gpu(x, args.gpu)
            t = cuda.to_gpu(t, args.gpu)

        optimizer.update(model, x, t)
        sum_loss += float(model.loss.data)
        sum_accuracy += float(model.accuracy.data) * len(t)
        train_count += len(t)

        print('loss %.5f, acc %.2f%%' % (sum_loss/train_count, 100*sum_accuracy/train_count))

        if train_iter.is_new_epoch:
            print('epoch: ', train_iter.epoch)
            print('train mean loss: %.5f, accuracy: %.2f%%' % (
                sum_loss / train_count, 100 * sum_accuracy / train_count))
            train_acc.append(sum_accuracy / train_count)
            # evaluation
            sum_accuracy = 0
            sum_loss = 0
            train_count = 0
            model.predictor.train = False
            for batch in dev_iter:
                batch = np.array(batch)
                x = np.vstack(batch[:,0])
                t = np.hstack(batch[:,1])
                # x, t = concat_examples(batch)

                if args.gpu >= 0:
                    x = cuda.to_gpu(x, args.gpu)
                    t = cuda.to_gpu(t, args.gpu)

                loss = model(x, t)
                sum_loss += float(loss.data)
                sum_accuracy += float(model.accuracy.data) * len(t)
                dev_count += len(t)

            dev_iter.reset()
            model.predictor.train = True
            print('dev mean  loss: %.5f, accuracy: %.2f%%' % (
                sum_loss / dev_count, 100 * sum_accuracy / dev_count))
            dev_acc.append(sum_accuracy / dev_count)
            # if train_iter.epoch > 1 and dev_acc[-1] > dev_acc[-2] < 0.001:
            #     print('improvement is too small')
            #     break

            sum_accuracy = 0
            sum_loss = 0
            dev_count = 0

    # Save the model and the optimizer
    print('save the model')
    serializers.save_npz(os.path.join(args.out, 'model'), model)
    print('save the optimizer')
    serializers.save_npz(os.path.join(args.out, 'state'), optimizer)

    print('save accuracy')
    import pickle
    with open(os.path.join(args.out, 'train_dev_acc'), 'wb') as f:
        pickle.dump({'train': train_acc, 'dev': dev_acc}, f)


if __name__ == '__main__':
    main()
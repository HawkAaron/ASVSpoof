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
from data_loader import DataSet, load_data
from model import *

def flatten(data):
    '''
    'data': (wavs, feats, dim), 'label': (wavs)
    '''
    label = []
    for i, wav in enumerate(data['data']):
        label += [data['label'][i]] * len(wav)
    data['label'] = label
    data['data'] = np.vstack(data['data'])

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=200,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
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
    model = L.Classifier(DNN(args.unit)) #MLP(args.unit, 2))
    if args.gpu >= 0:
        # Make a speciied GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    # Load the MNIST dataset
    # train, test = chainer.datasets.get_mnist()  # tuple_dataset.TupleDataset(images, labels)
    # feat_train = load_cqcc('train')
    # flatten(feat_train)
    # train = tuple_dataset.TupleDataset(feat_train['data'], feat_train['label'])
    # feat_test = load_cqcc('dev')
    # flatten(feat_test)
    # test = tuple_dataset.TupleDataset(feat_test['data'], feat_test['label'])

    # extract all feature
    train_data, train_label, _ = load_data()
    train_data = np.vstack(train_data)
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    train_data = (train_data - mean) / std
    test_data, test_label, _ = load_data(mode='dev')
    test_data = np.vstack(test_data)
    mean = np.mean(test_data, axis=0)
    std = np.std(test_data, axis=0)
    test_data = (test_data - mean) / std

    train = DataSet(train_data, np.hstack(train_label))
    test = DataSet(np.vstack(test_data), np.hstack(test_label))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize, n_processes=1)
    # test_iter = chainer.iterators.MultiprocessIterator(test, args.batchsize, n_processes=1, repeat=False, shuffle=False)

    sum_accuracy = 0
    sum_loss = 0
    train_count = 0
    test_count = 0

    train_acc = []
    test_acc = []

    while train_iter.epoch < args.epoch:
        batch = train_iter.next()   # feature: (batchsize, frames, window*dim)
        # Reduce learning rate by 0.5 every 25 epochs.
        if train_iter.epoch % 10 == 0 and train_iter.is_new_epoch:
            optimizer.lr *= 0.5
            print('Reducing learning rate to: ', optimizer.lr)

        x, t = concat_examples(batch)
        # batch = np.array(batch)
        # x = np.vstack(batch[:, 0])  # (frames, window*dim)
        # t = np.hstack(batch[:, 1])

        # random shuffle
        # indix = [i for i in range(len(t))]
        # np.random.shuffle(indix)
        # x = x[indix]
        # t = t[indix]

        if args.gpu >= 0:
            x = cuda.to_gpu(x, args.gpu)
            t = cuda.to_gpu(t, args.gpu)

        optimizer.update(model, x, t)
        sum_loss += float(model.loss.data)
        sum_accuracy += float(model.accuracy.data) * len(t)
        train_count += len(t)
        # print('loss ', sum_loss / train_count, ' acc', sum_accuracy / train_count)
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
            for batch in test_iter:

                x = np.vstack(batch[0])
                t = np.hstack(batch[1])
                # x, t = concat_examples(batch)

                if args.gpu >= 0:
                    x = cuda.to_gpu(x, args.gpu)
                    t = cuda.to_gpu(t, args.gpu)

                loss = model(x, t)
                sum_loss += float(loss.data)
                sum_accuracy += float(model.accuracy.data) * len(t)
                test_count += len(t)

            test_iter.reset()
            model.predictor.train = True
            print('test mean  loss: %.5f, accuracy: %.2f%%' % (
                sum_loss / test_count, 100 * sum_accuracy / test_count))
            test_acc.append(sum_accuracy / test_count)
            # if train_iter.epoch > 1 and test_acc[-1] > test_acc[-2] < 0.001:
            #     print('improvement is too small')
            #     break

            sum_accuracy = 0
            sum_loss = 0
            test_count = 0

    # Save the model and the optimizer
    print('save the model')
    serializers.save_npz(os.path.join(args.out, 'model'), model)
    print('save the optimizer')
    serializers.save_npz(os.path.join(args.out, 'state'), optimizer)

    print('save accuracy')
    import pickle
    with open(os.path.join(args.out, 'train_test_acc'), 'wb') as f:
        pickle.dump({'train': train_acc, 'test': test_acc}, f)


if __name__ == '__main__':
    main()
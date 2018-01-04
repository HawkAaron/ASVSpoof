import argparse, os
from models.common import *
from data_loader import DataSet, load_data, DataSetOnLine

parser = argparse.ArgumentParser(description='ASVSpoof')
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--dir', '-d', default='dnn', help='directory where model is, and save score')
parser.add_argument('--model', '-m', default='model_final', help='model to predict')
args = parser.parse_args()

model = None
chainer.serializers.load_npz(args.model, model)
model = model.predictor

if args.gpu >= 0:
    # Make a speciied GPU current
    chainer.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()  # Copy the model to the GPU

dev = DataSetOnLine(mode='dev', feat_type='fft', buf=False)
test = DataSetOnLine(mode='eval', feat_type='fft', buf=False)

dev_iter = chainer.iterators.MultiprocessIterator(dev, 1, n_prefetch=2, shared_mem=20*1024*1024, repeat=False, shuffle=False)
test_iter = chainer.iterators.MultiprocessIterator(test, 1, n_prefetch=2, shared_mem=20*1024*1024, repeat=False, shuffle=False)

def predict(data_iter, save_path):
    for batch in data_iter:
        x = np.vstack(batch[0])

        if args.gpu >= 0:
            x = cuda.to_gpu(x, args.gpu)

        y = model(x)

predict(dev_iter, 'score_dev')
predict(test_iter, 'score_test')
       
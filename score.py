import argparse, os
import models
from models.common import *
from data_loader import DataSet, load_data, DataSetOnLine

parser = argparse.ArgumentParser(description='ASVSpoof')
parser.add_argument('--batchsize', '-b', type=int, default=1, help='Number of images in each mini-batch')
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--dir', '-d', default='dnn', help='directory where model is, and save score')
parser.add_argument('--name', '-n', default='model_final', help='model name')
parser.add_argument('--model', '-m', default='DNN', help='Nnet model structure')
parser.add_argument('--feat', '-f', default='db4', help='feature type')
args = parser.parse_args()

# Set up a neural network to train
try:
    model = L.Classifier(getattr(models, args.model)())
except Exception as e:
    print(e)
    exit() 

chainer.serializers.load_npz(os.path.join(args.dir, args.name), model)
model = model.predictor

if args.gpu >= 0:
    # Make a speciied GPU current
    chainer.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()  # Copy the model to the GPU

dev = DataSetOnLine(mode='dev', feat_type=args.feat, buf=False)
test = DataSetOnLine(mode='eval', feat_type=args.feat, buf=False)

dev_iter = chainer.iterators.MultiprocessIterator(dev, args.batchsize, n_prefetch=2, shared_mem=20*1024*1024, repeat=False, shuffle=False)
test_iter = chainer.iterators.MultiprocessIterator(test, args.batchsize, n_prefetch=2, shared_mem=20*1024*1024, repeat=False, shuffle=False)

def convert_batch(batch, device=None):
    batch = np.array(batch)
    length_list = [len(mini[1]) for mini in batch]
    x = np.vstack(batch[:,0])
    # y = np.hstack(batch[:,1])
    t = np.hstack(batch[:,2]).tolist()

    if device is None:
        return (x, t, length_list)
    if device >= 0:
        x = cuda.to_gpu(x, device)
        # y = cuda.to_gpu(y, device)
        return (x, t, length_list)

def score_to_file(score, flist, fname):
    '''
    save score to fname
    '''
    with open(os.path.join(args.dir, fname), 'w') as f:
        for name, data in zip(flist, score):
            f.write('{} {}\n'.format(name, data))

def predict(data_iter, save_path):
    score = []
    flist = []
    for batch in data_iter:
        x, t, length_list = convert_batch(batch, args.gpu)
        y = model(x)
        cur = 0
        for l, name in zip(length_list, t):
            m = np.mean(y.data[cur:cur+l], axis=0)
            score.append(m[0] - m[1])
            flist.append(name)
            cur += l
            print(flist[-1], score[-1])

    score_to_file(score, flist, save_path)

print('dev score')
predict(dev_iter, 'score_dev')
print('eval score')
predict(test_iter, 'score_test')

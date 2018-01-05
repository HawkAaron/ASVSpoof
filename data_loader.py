import os, pickle
import numpy as np
from extract_feature import extract
import configparser

settings = configparser.ConfigParser()
settings.read('config')
ROOT = settings.get('data', 'root')
TRAIN = ROOT + 'ASVspoof2017_train/'
DEV = ROOT + 'ASVspoof2017_dev/'
EVAL = ROOT + 'ASVspoof2017_eval/'

PROTOCAL = os.path.join(ROOT, 'protocol/')
TP = os.path.join(PROTOCAL, 'ASVspoof2017_train.trn.txt')
DP = os.path.join(PROTOCAL, 'ASVspoof2017_dev.trl.txt')
EP = os.path.join(PROTOCAL, 'ASVspoof2017_eval_v2_key.trl.txt')

MODE = {'train': TP, 'dev': DP, 'eval': EP}
WAV = {'train': TRAIN, 'dev': DEV, 'eval': EVAL}
LABEL = {'genuine': 0, 'spoof': 1}


def load_data(mode='train', feat_type='db4', update=False, fresh=False):
    if fresh:
        return load_all_feature(mode, feat_type)
    
    save_path = mode + '.' + feat_type
    if not update and os.path.isfile(save_path):
        with open(save_path, 'rb') as f:
            return pickle.load(f)
    data = load_all_feature(mode, feat_type)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    return data

def feat_window(feat):
    window = 6  # actual size = 2 * window - 1
    feat = np.pad(feat, [[0, 0], [window - 1, window - 1]], mode='edge')
    tmp_feat = []
    for i in range(window - 1, feat.shape[1] - window, 2*window-1):
        fram_win = feat[:, i-window+1: i+window].reshape(-1) # store feat_dim * 11 map as vector
        tmp_feat.append(fram_win)
    return np.array(tmp_feat, dtype=np.float32)

def feat_padding(feat, width=400):
    dim = feat.shape[0]

    padding = width - feat.shape[1] % width
    if padding < 200:
        feat = np.pad(feat, [[0, 0], [0, padding]], mode='edge')
    if feat.shape[1] % width:
        feat = feat[:, :-(feat.shape[1]%width)]
    tmp_feat = feat.T.reshape(-1, width, dim)

    return tmp_feat

def load_all_feature(mode='train', feat_type='db4'):
    flist = []  # wav file list
    label = []  # wav label list
    with open(MODE[mode], 'r') as f:
        for line in f:
            data = line.split()
            flist.append(data[0])
            label.append(LABEL[data[1]])

    final_flist = []
    final_label = []
    final_feat = []
    for idx in range(len(flist)):
        wav_path = os.path.join(WAV[mode], flist[idx])
        try:
            feat = extract(wav_path, feat_type)
        except:
            print('CORRUPTED WAV: ', wav_path)
            continue

        if feat_type == 'fft':
            feat = feat_padding(feat)
        if feat_type == 'db4':
            feat = feat_window(feat)

        final_feat.append(feat)
        final_flist.append(flist[idx])
        final_label.append([label[idx]] * len(feat)) # label expand (wavs, frames)
    return final_feat, final_label, final_flist

class DataSet():
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if type(idx) is not slice:
            return (self.data[idx], self.label[idx])
        
        # slice reading
        items = []
        length = self.__len__()
        for i in range(idx.start, idx.stop):
            # if i >= self.__len__():
            #     print('FUCKING INDEX OUT OF RANGE: {} {}'.format(idx.start, idx.stop))
            #     break
            i %= length
            items.append((self.data[i], self.label[i]))
        return items
        
class DataSetOnLine():
    '''
    Online data loader
    '''
    def __init__(self, mode='train', feat_type='db4', buf=True):
        '''
        mode should be `train`, `dev`, or `eval`
        '''
        self.mode = mode
        self.feat_type = feat_type  # wav feature type
        flist = []  # wav file list
        label = []  # wav label list
        with open(MODE[mode], 'r') as f:
            for line in f:
                data = line.split()
                flist.append(data[0])
                label.append(LABEL[data[1]])
        # shuffle wavs
        # indices = np.arange(len(flist))
        # np.random.shuffle(indices)
        # self.flist = flist[indices]
        # self.label = label[indices]

        self.flist = flist
        self.label = label
        self.buf = buf
        self.buffer = [None] * len(self.label)  # buffer save read wavs

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        '''
        Return:
            `tuple(feature, label, file name)`
            `feature: (wav, feats, dim*11)
        **WAV CORRUPT EXCEPTION NOT HUNDLE**
        '''
        if type(idx) is not slice:
            if self.buf and self.buffer[idx] is not None:
                return self.buffer[idx]

            wav_path = os.path.join(WAV[self.mode], self.flist[idx])
            feat = extract(wav_path, self.feat_type)   # wav corrupt not hundle

            if self.feat_type == 'fft':
                feat = feat_padding(feat)
            else:
                feat = feat_window(feat)

            item = (feat, [self.label[idx]] * feat.shape[0], self.flist[idx])
            if self.buf:
                self.buffer[idx] = item
            return item

        # slice reading
        items = []
        length = self.__len__()
        for i in range(idx.start, idx.stop):
            i %= length
            
            if self.buf and self.buffer[i] is not None:
                items.append(self.buffer[i])
                continue
            wav_path = os.path.join(WAV[self.mode], self.flist[i])
            feat = extract(wav_path, self.feat_type)

            if self.feat_type == 'fft':
                feat = feat_padding(feat)
            else:
                feat = feat_window(feat)

            items.append((feat, [self.label[i]] * feat.shape[0], self.flist[i]))
            if self.buf:
                self.buffer[i] = items[-1]

        return items

if __name__ == '__main__':
    train = DataSetOnLine('train', 'fft', False)
    print(train[3])
    # load_all_feature('train', 'fft')
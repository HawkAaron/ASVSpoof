import os, pickle
import numpy as np
from extract_feature import extract

ROOT = '/leuvenfs/workspace/course/asr/project2/'
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

        window = 6  # actual size = 2 * window - 1
        feat = np.pad(feat, [[0, 0], [window - 1, window - 1]], mode='edge')
        tmp_feat = []
        for i in range(window - 1, feat.shape[1] - window, 2*window-1):
            fram_win = feat[:, i-window+1: i+window].reshape(-1) # store feat_dim * 11 map as vector
            tmp_feat.append(fram_win)
        tmp_feat = np.array(tmp_feat, dtype=np.float32) # (frames, dim*11)
        final_feat.append(tmp_feat)
        final_flist.append(flist[idx])
        final_label.append([label[idx]] * len(tmp_feat)) # label expand (wavs, frames)
    return final_feat, final_label, final_flist

class DataSet():
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        **NOT IMPLEMENT FOR SLICE**
        '''
        return (self.data[idx], self.label[idx])

class DataSetOnLine():
    '''
    Online data loader
    **NOT FINISHED**
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
        if self.buf and self.buffer[idx] is not None:
            return self.buffer[idx]

        wav_path = os.path.join(WAV[self.mode], self.flist[idx])
        feat = extract(wav_path, self.feat_type)   # wav corrupt not hundle

        window = 6  # actual size = 2 * window - 1
        feat = np.pad(feat, [[0, 0], [window - 1, window - 1]], mode='edge')
        final_feat = []
        for i in range(window - 1, feat.shape[1] - window, window):
            tmp = feat[:, i-window+1: i+window].reshape(-1) # store feat_dim/window * 11 map as vector
            final_feat.append(tmp)
        final_feat = np.array(final_feat, dtype=np.float32)

        item = (final_feat, self.label[idx], self.flist[idx])
        if self.buf:
            self.buffer[idx] = item
        return item

if __name__ == '__main__':
    train = load_data('train')
    test = load_data('dev')
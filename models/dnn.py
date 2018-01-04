from common import *

class DNN(chainer.Chain):
    def __init__(self, h_dim=2048):
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
        h = F.dropout(h)

        h = self.l2(x)
        # h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h)

        h = self.l3(x)
        # h = self.bn3(h)
        h = F.relu(h)
        h = F.dropout(h)
        return self.l4(h)
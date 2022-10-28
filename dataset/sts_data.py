import pickle
from os.path import exists, join

import pandas as pd
from torch.utils.data import Dataset

from conf import ROOT


class STSDataset(Dataset):
    """
    ['s1', 's2', 'label']
    """

    def __init__(self, path):
        if not exists(path):
            path = join(ROOT, path)
        self.features = ['s1', 's2', 'label']
        self.data = self.load_csv_data(path)

    def __getitem__(self, item):
        sample = self.data[item]
        sample = {k: v for k, v in sample.items() if k in self.features}
        return sample

    def __len__(self):
        return len(self.data)

    def load_pickle_data(self, path):
        with open(path, 'rb') as fin:
            data = pickle.load(fin)
        assert list(data[0].keys()) == self.features
        return data

    def load_csv_data(self, path):
        df = pd.read_csv(path)
        df = df[self.features]
        data = df.to_dict('records')
        return data


if __name__ == '__main__':
    _d = STSDataset('../data/session/intent_line6_sts.csv')

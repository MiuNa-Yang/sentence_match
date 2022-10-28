import pandas as pd
from torch.utils.data import Dataset


class ClsDataset(Dataset):
    def __init__(self, path):

        df = pd.read_csv(path)
        df.rename(columns={"text_a": "text"}, inplace=True)
        self.data = df.to_dict('records')

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    ds = ClsDataset("../data/train.csv")
    sample = next(iter(ds))
    print(sample)

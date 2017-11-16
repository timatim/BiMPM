import torch
import pandas as pd
from torch.utils.data import Dataset
import utils


class QuoraDatum():
    def __init__(self, p, q, label, words, chars, cuda=False):
        self.p = p
        self.q = q
        self.p_words, self.p_chars = utils.sentence_to_padded_index_sequence(p, words, chars, cuda=cuda)
        self.q_words, self.q_chars = utils.sentence_to_padded_index_sequence(q, words, chars, cuda=cuda)
        self.label = label


class QuoraDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        d = self.data[key]

        return (d.p_words, d.p_chars, d.q_words, d.q_chars), d.label


def make_dataloader(df, words, chars, batch_size=128, shuffle=True, cuda=False):
    dataset = QuoraDataset([QuoraDatum(row['p'], row['q'], row['label'], words, chars, cuda=cuda)
                            for (_, row) in df.iterrows()])
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)
    return train_loader
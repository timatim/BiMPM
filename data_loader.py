import torch
import pandas as pd
from torch.utils.data import Dataset
import utils


class QuoraDataset(Dataset):
    def __init__(self, p, q, label, words, chars, seq_len=50, word_len=20, cuda=False):
        self.p = p
        self.q = q
        self.label = label
        self.words = words
        self.chars = chars
        self.seq_len = seq_len
        self.word_len = word_len
        self.cuda = cuda

    def __len__(self):
        return len(self.label)

    def __getitem__(self, key):
        p_words, p_chars = utils.sentence_to_padded_index_sequence(
            self.p[key], self.words, self.chars,
            seq_len=self.seq_len, word_len=self.word_len, cuda=self.cuda)

        q_words, q_chars = utils.sentence_to_padded_index_sequence(
            self.q[key], self.words, self.chars,
            seq_len=self.seq_len, word_len=self.word_len, cuda=self.cuda)

        return (p_words, p_chars, q_words, q_chars), self.label[key]


def make_dataloader(df, words, chars, seq_len=50, word_len=20, batch_size=128, shuffle=True, cuda=False):
    dataset = QuoraDataset(df['p'], df['q'], df['label'], words, chars, seq_len=seq_len, word_len=word_len, cuda=cuda)
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)
    return train_loader
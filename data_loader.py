import torch
import pandas as pd
from torch.utils.data import Dataset
import utils


class QuoraDataset(Dataset):
    def __init__(self, p, q, label, words, chars, seq_len=50, word_len=20, cuda=False):
        """
        Initializes a QuoraDataset object, subclass of torch Dataset
        :param p: list-like of passage strings
        :param q: list-like of passage strings
        :param label: list-like of binary labels [0,1]
        :param words: 
        :param chars: 
        :param seq_len: 
        :param word_len: 
        :param cuda: 
        """
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
        return (self.p[key], self.q[key]), self.label[key]

    def collate_batch(self, batch):
        labels = [int(b[1]) for b in batch]

        p_sentences = [b[0][0].split() for b in batch]
        q_sentences = [b[0][1].split() for b in batch]

        # get longest seq_len in batch, pad to seq_len
        max_seq_len = min(max(max([len(p) for p in p_sentences]), max([len(q) for q in q_sentences])), self.seq_len)

        p_words_chars = [utils.sentence_to_padded_index_sequence(
            p, self.words, self.chars, seq_len=max_seq_len, word_len=self.word_len, cuda=self.cuda)
            for p in p_sentences
        ]
        p_words = torch.stack([p[0] for p in p_words_chars])
        p_chars = torch.stack([p[1] for p in p_words_chars])

        q_words_chars = [utils.sentence_to_padded_index_sequence(
            q, self.words, self.chars, seq_len=max_seq_len, word_len=self.word_len, cuda=self.cuda)
            for q in q_sentences
        ]
        q_words = torch.stack([q[0] for q in q_words_chars])
        q_chars = torch.stack([q[1] for q in q_words_chars])

        return (p_words, p_chars, q_words, q_chars), torch.LongTensor(labels)


def make_dataloader(df, words, chars, seq_len=50, word_len=20, batch_size=128, shuffle=True, cuda=False):
    """
    Returns a pytorch DataLoader of the Quora dataset
    :param df: a pandas DataFrame-like with columns ['p', 'q', 'label']
    :param words: dictionary of vocabs
    :param chars: dictionary of character vocabs
    :param seq_len: sequence length to pad to
    :param word_len: word length to pad to
    :param batch_size: 
    :param shuffle: 
    :param cuda: 
    :return: a pytorch DataLoader object
    """
    dataset = QuoraDataset(df['p'], df['q'], df['label'], words, chars, seq_len=seq_len, word_len=word_len, cuda=cuda)
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               collate_fn=dataset.collate_batch,
                                               shuffle=shuffle)
    return train_loader

import torch
import torch.nn as nn
import numpy as np
import os


class WordRepresentationLayer(nn.Module):
    """
    Word Representation Layer of model. Each word is represented by its word embedding
    weight concatenated with the hidden state of the character level LSTM as the character
    composed embedding.
    """

    def __init__(self, embedding_weights, vocabs, char_vocabs, word_dim=300, batch_size=128, char_lstm_dim=50,
                 char_dim=20, dropout=0.1):
        """
        Initializes the WordRepresentationLayer
        :param vocab_size: # of vocabs in word embedding
        :param embedding_dim: word embedding dimension
        :param batch_size: batch size
        :param charLSTM_dim: hidden state dimension of the char-level LSTM (also the character composed embedding dim)
        :param char_dim: dimension of character embedding
        """
        super(WordRepresentationLayer, self).__init__()
        # initialize word embedding
        self.word_embedding = nn.Embedding(len(vocabs), word_dim, padding_idx=len(vocabs)-1)
        self.word_embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
        # fixed pre-trained embedding
        self.word_embedding.weight.requires_grad = False
        self.word_dim = word_dim
        self.word2idx = vocabs

        # initialize character composed embedding with lstm
        self.char_embedding = nn.Embedding(len(char_vocabs), char_dim, padding_idx=len(char_vocabs)-1)
        self.char_LSTM = nn.LSTM(char_dim, char_lstm_dim, dropout=dropout)
        self.char_dim = char_dim
        self.char_lstm_dim = char_lstm_dim
        self.char2idx = char_vocabs

        self.batch_size = batch_size
        self.embedding_dim = word_dim + char_lstm_dim

    def forward(self, words, chars):
        """
        Forward function of the WordRepresentationLayer that computes the representation of the words
        :param words: indices of words in the sentence [batch, seq_len]
        :param chars: indices of chars in the sentences [batch, seq_len, word_len]
        :return: representations of words in the sentence [batch, seq_len, word_dim]
        """
        batch_size, seq_len, word_len = chars.size()
        chars = chars.view(batch_size*seq_len, word_len)
        chars = chars.permute(1, 0)
        words = words.permute(1, 0)

        word_embs = self.word_embedding(words)

        # pass to LSTM, and take final hidden state of LSTM. Reshape to batch_size * seq_len
        char_embs = self.char_embedding(chars)
        char_composed_word_embs = self.char_LSTM(char_embs)[0][-1].view(batch_size, seq_len, -1).permute(1, 0, 2)

        # concatenate results
        word_rep = torch.cat([word_embs, char_composed_word_embs], dim=2)
        return word_rep


if __name__ == '__main__':
    print("Testing WordRepresentationLayer")

    test_strings = ["How do I get funding for my web based startup idea ?",
                    "How do I get seed funding pre product ?"]

    # load embedding
    embedding_dir = '../embedding'
    words_to_load = 10000
    with open(os.path.join(embedding_dir, 'glove.840B.300d.txt')) as f:
        loaded_embeddings = []
        words = {}
        chars = {}
        idx2words = {}
        ordered_words = []
        valid_count = 0
        for i, line in enumerate(f):
            if valid_count >= words_to_load:
                break
            s = line.split()
            # check for words with spaces?
            if len(s) != 301:
                continue
            loaded_embeddings.append(np.asarray(s[1:]))
            # collect char
            for c in s[0]:
                if c not in chars:
                    chars[c] = len(chars)

            words[s[0]] = valid_count
            idx2words[i] = s[0]
            ordered_words.append(s[0])
            valid_count += 1

    loaded_embeddings = np.array(loaded_embeddings).astype(float)
    print("Loaded word embedding shape: %s" % (loaded_embeddings.shape,))

    test_data = [s.split() for s in test_strings]
    # test_data = [[words[word] for word in sentence if word in words] for sentence in test_data]
    # initialize WordRepresentationLayer
    wr = WordRepresentationLayer(loaded_embeddings, words, chars)
    print(wr(test_data[0]).size())
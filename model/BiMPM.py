import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from . import WordRepresentationLayer, MatchingLayer
from time import time


class PredictionLayer(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=100, output_dim=2, dropout=0.1):
        super(PredictionLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.hidden(x.view(-1, self.input_dim))
        x = F.softmax(self.dropout(self.output(x)))
        return x


class BiMPM(nn.Module):
    def __init__(self, pretrained_embedding, vocabs, char_vocabs,
                 word_emb_dim=300, char_emb_dim=50, char_dim=20,
                 context_emb_dim=100, context_dropout=0.1,
                 aggregation_dim=100, aggregation_dropout=0.1,
                 prediction_dim=100,
                 output_dim=2, perspectives=5):
        super(BiMPM, self).__init__()
        self.word_layer = WordRepresentationLayer(pretrained_embedding, vocabs, char_vocabs,
                                                  word_dim=word_emb_dim, char_lstm_dim=char_emb_dim, char_dim=char_dim,
                                                  dropout=0.1)
        self.context_layer = nn.LSTM(word_emb_dim + char_emb_dim, context_emb_dim, #batch_first=True,
                                     bidirectional=True, dropout=context_dropout)
        self.matching_layer = MatchingLayer(perspectives=perspectives)
        self.aggregation_layer = nn.LSTM(2 * perspectives, aggregation_dim,
                                         bidirectional=True, dropout=aggregation_dropout)
        # self.aggregation_layer = nn.LSTM(2 * context_emb_dim, aggregation_dim, #batch_first=True,
        #                                  bidirectional=True, dropout=aggregation_dropout)
        self.aggregation_dim = aggregation_dim
        self.prediction_layer = PredictionLayer(input_dim=4*aggregation_dim,
                                                hidden_dim=prediction_dim,
                                                output_dim=output_dim)

    def forward(self, data):
        p_words, p_chars, q_words, q_chars = data
        p, q = self.word_layer(p_words, p_chars), self.word_layer(q_words, q_chars)
        p, q = self.context_layer(p)[0], self.context_layer(q)[0]
        p, q = self.matching_layer(p, q)
        p, q = self.aggregation_layer(p)[0], self.aggregation_layer(q)[0]
        output = self.prediction_layer(torch.cat([p[-1, :, :self.aggregation_dim],
                                                  p[0, :, self.aggregation_dim:],
                                                  q[-1, :, :self.aggregation_dim],
                                                  q[0, :, self.aggregation_dim:]], dim=-1))
        return output

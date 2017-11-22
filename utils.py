import numpy as np
import torch


def word_to_padded_index_sequence(word, chars, is_padding=True, word_len=20, cuda=False):
    """
    
    :param word: word to be tokenized by characters
    :param padding: if the word is a padding word itself, return all paddings
    """
    PADDING = "<PAD>"
    UNKNOWN = "<UNK>"

    if is_padding:
        return [chars[PADDING]] * word_len

    char_indices = []
    for i in range(word_len):
        if i < len(word):
            if word[i] in chars:
                index = chars[word[i]]
            else:
                index = chars[UNKNOWN]
        else:
            index = chars[PADDING]
        char_indices.append(index)

    if cuda:
        char_indices = torch.cuda.LongTensor(char_indices)
    else:
        char_indices = torch.LongTensor(char_indices)
    return char_indices


def sentence_to_padded_index_sequence(sentence, words, chars, seq_len=50, word_len=20, cuda=False):
    """
    Converts tokenized sentences to padded word indices of specified seq_len and each word
    in the sentence to character indices
    :param sentence: list of words
    :param words: word vocabulary
    :param chars: character vocabulary
    :param seq_len: padded length
    :param word_len: padded word length
    :param cuda: whether or not to use cuda tensors
    :return: (word indices, character indices)
    """

    PADDING = "<PAD>"
    UNKNOWN = "<UNK>"

    sentence_words = []

    # list of each word's character indices list
    words_as_chars = []
    tokens = sentence.lower().split()

    for i in range(seq_len):
        # character indices of each word
        word_as_chars = []

        if i < len(tokens):
            if tokens[i] in words:
                index = words[tokens[i]]
            else:
                index = words[UNKNOWN]
            word_as_chars.append(word_to_padded_index_sequence(tokens[i], chars, cuda=cuda))
        else:
            index = words[PADDING]
            word_as_chars.append(word_to_padded_index_sequence('', chars, word_len=word_len,
                                                               cuda=cuda, is_padding=True))

        if cuda:
            words_as_chars.append(torch.cuda.LongTensor(word_as_chars))
        else:
            words_as_chars.append(torch.LongTensor(word_as_chars))

        sentence_words.append(index)

    if cuda:
        sentence_words = torch.cuda.LongTensor(sentence_words)
    else:
        sentence_words = torch.LongTensor(sentence_words)
    return sentence_words, torch.stack(words_as_chars).squeeze()


def load_embedding(embedding_path, words_to_load=1000000):

    with open(embedding_path) as f:
        loaded_embeddings = []
        words = {}
        chars = {}
        idx2words = {}
        ordered_words = []

        for i, line in enumerate(f):
            if len(words) >= words_to_load:
                break
            s = line.split()
            # check for words with spaces?
            if len(s) != 301:
                continue
            # check if already loaded?
            if s[0] in words:
                continue
            loaded_embeddings.append(np.asarray(s[1:]))
            # collect char
            for c in s[0]:
                if c not in chars:
                    chars[c] = len(chars)
            words[s[0]] = len(words)
            idx2words[i] = s[0]
            ordered_words.append(s[0])

    # add unknown to word and char
    loaded_embeddings.append(np.random.rand(300))
    words["<UNK>"] = len(words)

    # add padding
    loaded_embeddings.append(np.zeros(300))
    words["<PAD>"] = len(words)

    chars["<UNK>"] = len(chars)
    chars["<PAD>"] = len(chars)

    loaded_embeddings = np.array(loaded_embeddings).astype(float)

    return loaded_embeddings, words, chars

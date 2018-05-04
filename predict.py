import torch
from torch.autograd import Variable
import numpy as np
import argparse
from model.BiMPM import BiMPM
import gc
import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch model on Quora Paraphrase Detection")
    parser.add_argument('--embedding', type=str, default='./embedding/wordvec.txt', help='path to embedding')
    parser.add_argument('--vocab', type=int, default=400000, help='words to load from embedding')
    parser.add_argument('--model', type=str, default='./models/model.pth', help='path to model dict file')
    parser.add_argument('--data', type=str, default='./data/test.tsv', help='path to test data')
    parser.add_argument('--seq-len', type=int, default=50, help='length to pad sentences to')
    parser.add_argument('--word-len', type=int, default=20, help='length to pad words to')
    parser.add_argument('--perspectives', type=int, default=5, help='number of perspectives to use in matching')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')

    args = parser.parse_args()
    print(args)

    # set random seed
    torch.manual_seed(args.seed)


    print("Loading pre-trained embedding...")
    loaded_embedding, words, chars = utils.load_embedding(args.embedding, words_to_load=args.vocab)
    print("Embedding size: %s" % (loaded_embedding.shape,))

    gc.collect()

    p_sentences = [
        "What is the most effective way to break a porn addiction ?",
        "Can we do affiliate marketing without having a website or a company ?",
        "How do I restore my self confidence ?",
        "hi there",
        "Is honey a viable alternative to sugar for diabetics ?"
    ]

    q_sentences = [
        "What ` s the best way to get rid of porn addiction ?",
        "How do I do affiliate marketing without a website ?",
        "What should I do to restore my self confidence and faith in myself ?",
        "hello there",
        "How would you compare the United States ' euthanasia laws to Denmark ?"
    ]

    p_sentences = [p.lower().split() for p in p_sentences]
    q_sentences = [q.lower().split() for q in q_sentences]

    max_seq_len = 20  # max(len(p_sentences[0]), len(q_sentences[0]))
    word_len = 15

    p_words_chars = [utils.sentence_to_padded_index_sequence(
        p, words, chars, seq_len=max_seq_len, word_len=word_len, cuda=False)
        for p in p_sentences
    ]
    p_words = torch.stack([p[0] for p in p_words_chars])
    p_chars = torch.stack([p[1] for p in p_words_chars])

    q_words_chars = [utils.sentence_to_padded_index_sequence(
        q, words, chars, seq_len=max_seq_len, word_len=word_len, cuda=False)
        for q in q_sentences
    ]

    q_words = torch.stack([q[0] for q in q_words_chars])
    q_chars = torch.stack([q[1] for q in q_words_chars])

    print("Loading model...")
    model = BiMPM(loaded_embedding, words, chars, perspectives=args.perspectives)
    model.load_state_dict(torch.load(args.model, lambda storage, loc: storage))
    model.eval()

    del loaded_embedding
    gc.collect()

    if args.cuda:
        model.cuda()
    print(model)

    result = model((Variable(p_words), Variable(p_chars), Variable(q_words), Variable(q_chars))).data.numpy()
    print(result)
    print(np.argmax(result, axis=1))

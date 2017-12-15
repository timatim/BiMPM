import torch
from torch.autograd import Variable
import pandas as pd
import argparse
import os
from model.BiMPM import BiMPM
import gc
import data_loader
import utils
from time import time


def test_model(loader, model):
    """
    Help function that tests the models's performance on a dataset
    :param: loader: data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for data, labels in loader:
        data = [Variable(item) for item in data]
        if args.cuda:
            data = [d.cuda() for d in data]
            labels = labels.cuda()
        outputs = model(data)
        predicted = (outputs.max(1)[1].data.long()).view(-1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    model.train()
    return 100 * correct / total


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
    parser.add_argument('--batch-size', type=int, default=16)

    args = parser.parse_args()
    print(args)

    # set random seed
    torch.manual_seed(args.seed)

    batch_size = args.batch_size

    print("Loading pre-trained embedding...")
    loaded_embedding, words, chars = utils.load_embedding(args.embedding, words_to_load=args.vocab)
    print("Embedding size: %s" % (loaded_embedding.shape,))
    print("Preparing data loader...")
    # prepare data
    quora_test = pd.read_csv(args.data, sep='\t', names=['label', 'p', 'q', 'id'])

    test_loader = data_loader.make_dataloader(quora_test,
                                               words, chars, batch_size=batch_size,
                                               seq_len=args.seq_len, word_len=args.word_len, cuda=args.cuda)

    gc.collect()

    print("Loading model...")
    model = BiMPM(loaded_embedding, words, chars, perspectives=args.perspectives)
    model.load_state_dict(torch.load(args.model, lambda storage, loc: storage))
    del loaded_embedding
    gc.collect()

    if args.cuda:
        model.cuda()
    print(model)

    print("Testing...")
    test_acc = test_model(test_loader, model)

    print("Accuracy: %.4f", test_acc)

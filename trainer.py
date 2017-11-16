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
    return (100 * correct / total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch model on Quora Paraphrase Detection")
    parser.add_argument('--embedding', type=str, default='./embedding/wordvec.txt', help='path to embedding')
    parser.add_argument('--vocab', type=int, default=400000, help='words to load from embedding')
    parser.add_argument('--data', type=str, default='./data/', help='location of train.tsv, dev.tsv, test.tsv')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--save', type=str, default='./models/', help='path to store models')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    batch_size = args.batch_size
    num_epochs = args.epochs

    print("Loading pre-trained embedding...")
    loaded_embedding, words, chars = utils.load_embedding(args.embedding, words_to_load=args.vocab)
    print("Embedding size: %s" % (loaded_embedding.shape,))
    print("Preparing data loader...")
    # prepare data
    quora_train = pd.read_csv(os.path.join(args.data, 'train.tsv'), sep='\t', names=['label', 'p', 'q', 'id'])
    quora_dev = pd.read_csv(os.path.join(args.data, 'dev.tsv'), sep='\t', names=['label', 'p', 'q', 'id'])
    train_size = len(quora_train)

    train_loader = data_loader.make_dataloader(quora_train, words, chars, batch_size, cuda=args.cuda)
    dev_loader = data_loader.make_dataloader(quora_dev, words, chars, batch_size, cuda=args.cuda)

    del quora_train
    del quora_dev
    gc.collect()

    print("Defining models, loss function, optimizer...")
    # define models, loss, optimizer
    model = BiMPM(loaded_embedding, words, chars)
    del loaded_embedding
    gc.collect()

    if args.cuda:
        model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    print(model)

    print("beginning training...")
    # training the Model
    train_acc_history = []
    validation_acc_history = []
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_loader):
            if args.cuda:
                data = [d.cuda() for d in data]
                labels = labels.cuda()

            label_batch = Variable(labels)
            data = [Variable(item) for item in data]
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, label_batch)
            start = time()
            loss.backward()
            # print('backward', time() - start)
            optimizer.step()

            # report performance
            if i % 50 == 0:
                train_acc = test_model(train_loader, model)
                val_acc = test_model(dev_loader, model)
                print('Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4}, Train Acc: {5}, Validation Acc:{6}'.format(
                    epoch + 1, num_epochs, i + 1, len(train_loader), loss.data[0],
                    train_acc, val_acc))
                train_acc_history.append(train_acc)
                validation_acc_history.append(val_acc)

        torch.save(model.state_dict(), os.path.join(args.save, "BiMPM_%d.pth" % epoch))
    print("Train Accuracy:")
    print(train_acc_history)
    print("Validation Accuracy:")
    print(validation_acc_history)

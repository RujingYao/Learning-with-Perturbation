
import torch
import numpy as np
import pandas as pd
from torchtext import data, vocab


class MyDataset(data.Dataset):
    def __init__(self, csv_path, fields):

        examples = []
        csv_data = pd.read_csv(csv_path, header=None)

        for label, text, bias in zip(csv_data[0], csv_data[1], csv_data[2]):

            examples.append(data.Example.fromlist([label, text, bias], fields))

        super(MyDataset, self).__init__(examples, fields)


def data_iter(train_path,valid_path, test_path, batch_size, device, fields, TEXT):
    train = MyDataset(train_path, fields)
    valid = MyDataset(valid_path, fields)
    test = MyDataset(test_path, fields)
    vectors = vocab.Vectors(name='glove.6B.300d.txt', cache='vector_cache')
    TEXT.build_vocab(train, vectors=vectors)

    weight_matrix = TEXT.vocab.vectors
    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train, valid, test),
        batch_sizes=(batch_size, batch_size, batch_size),
        device=device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        shuffle = True,
        repeat=False)

    return train_iter, valid_iter, test_iter, weight_matrix


def text_token(text):
    return str(text).split(" ")

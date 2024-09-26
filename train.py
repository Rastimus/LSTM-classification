import os
import argparse
import pdb
from email.policy import default

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader

from Process_data import Corpus_Extr, Pad_sequences, PhraseDataset
from SentimentLSTM import SentimentRNN
from utils import *
import gc

def train(args, model, trains):
    optimizer = Adam(model.parameters(), lr=args.lr)
    model.train()
    counter = 0
    clip = 5
    print_every = 100
    losses = []
    accs = []
    for epoch in range(args.num_epochs):
        a = np.random.choice(len(trains) - 1, 1000)
        train_features = PhraseDataset(trains.loc[trains.index.isin(np.sort(a))], args.pad_sequences[a])
        train_loader = DataLoader(train_features, batch_size=args.batch_size, shuffle=True)
        running_loss = 0.0
        running_acc = 0.0
        h = model.init_hidden(32)
        for idx, (inputs, labels) in enumerate(train_loader):
            counter += 1
            gc.collect()
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            optimizer.zero_grad()
            if inputs.shape[0] != args.batch_size:
                break
            # get the output from the model
            output, h = model(inputs, h)
            labels = torch.nn.functional.one_hot(labels, num_classes=5)
            # calculate the loss and perform backprop
            # criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(output, labels)
            loss.backward()
            running_loss += loss.cpu().detach().numpy()
            running_acc += (output.argmax(dim=1) == labels.argmax(dim=1)).float().mean()            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if idx % 20 == 0:
                print("Epoch: {}/{}...".format(epoch + 1, args.num_epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format((running_loss / (idx + 1))))
                losses.append(float(running_loss / (idx + 1)))
                print(f'acc:{running_acc / (idx + 1)}')
                accs.append(running_acc / (idx + 1))
                torch.save(model.state_dict(), args.model_path)  # save the model


def test(args, model, tests):
    model.eval()  # 切换到评估模式
    a = np.random.choice(len(tests) - 1, 1000)
    test_features = PhraseDataset(tests.loc[tests.index.isin(np.sort(a))], args.pad_sequences[a])
    test_loader = DataLoader(test_features, batch_size=args.batch_size, shuffle=False)
    running_loss = 0.0
    running_acc = 0.0
    h = model.init_hidden(args.batch_size)  # 初始化隐藏状态

    criterion = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失

    with torch.no_grad():  # 禁用梯度计算
        for idx, (inputs, labels) in enumerate(test_loader):
            # 如果输入的批次大小与设定的不符，跳过该批次
            if inputs.shape[0] != args.batch_size:
                continue

            # 获取模型输出
            output, h = model(inputs, h)
            labels = torch.nn.functional.one_hot(labels, num_classes=5)  # 转换标签为one-hot形式

            # 计算损失
            loss = criterion(output, labels)
            running_loss += loss.cpu().detach().numpy()  # 累加损失

            # 计算准确率
            running_acc += (output.argmax(dim=1) == labels.argmax(dim=1)).float().mean()

    # 计算平均损失和准确率
    avg_loss = running_loss / len(test_loader)
    avg_acc = running_acc / len(test_loader)

    print(f'Test Loss: {avg_loss:.6f}, Test Accuracy: {avg_acc:.6f}')


def main():
    # Parse the cmd arguments
    trains = pd.read_csv('data/train.tsv',sep = '\t')
    tests = pd.read_csv('data/test.tsv',sep = '\t')
    corpus, vocab_to_int, phrase_to_int = Corpus_Extr(trains)
    parser = argparse.ArgumentParser()

    # Path and file configs
    parser.add_argument('--data_path', default='./data', help='The dataset path.', type=str)
    # parser.add_argument('--model_path', default='./model/bilstm.pt', help='The model will be saved to this path.', type=str)
    parser.add_argument('--train_file', default='train.tsv', type=str)
    # parser.add_argument('--val_file', default='val_data.txt', type=str)
    parser.add_argument('--test_file', default='test.tsv', type=str)
    parser.add_argument('--model_path', default='model/bilstm.pt', help='The model will be saved to this path.', type=str)

    # Model configs
    parser.add_argument('--embed_size', default=400, help='Tokens will be embedded to a vector.', type=int)
    parser.add_argument('--hidden_dim', default=256, help='The hidden state dim of sLSTM.', type=int)
    parser.add_argument('--output_size',default=5, help='The output size of sLSTM.', type=int)
    parser.add_argument('--n_layers', default=2, help='The number of layers.', type=int)
    parser.add_argument('--vocab_size', default=len(vocab_to_int), help='The size of vocab.', type=int)

    # Optimizer config
    parser.add_argument('--lr', default=1e-4, help='Learning rate of the optimizer.', type=float)

    # Training configs
    parser.add_argument('--batch_size', default=32, help='Batch size for training.', type=int)
    # parser.add_argument('--test_batch_size', default=16, help='Batch size for testing.', type=int)
    parser.add_argument('--num_epochs', default=20, help='Batch size for training.', type=int)
    # parser.add_argument('--eval_steps', default=200, help='Total number of training epochs to perform.', type=int)

    # Device config
    parser.add_argument('--gpu', default=0, type=int)

    # Mode config
    # parser.add_argument('--test', help='Test on the testset.', action='store_true')

    args = parser.parse_args()

    # Specify the device. If you has a GPU, the training process will be accelerated.
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    args.device = device

    # Every word must be mapped to a unique index
    args.vocab_size = len(vocab_to_int)
    args.corpus = corpus
    args.pad_sequences = Pad_sequences(phrase_to_int,30)



    model = SentimentRNN(args)
    model = model.to(args.device)   # move the model into GPU if GPU is available.

    train(args, model, trains)

    # model.load_state_dict(torch.load(args.model_path))
    # test(args, model, tests)



if __name__ == '__main__':
    main()
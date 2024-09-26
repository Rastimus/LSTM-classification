import os

import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from collections import Counter


def Corpus_Extr(df):
    print('构建语料库。。。')
    corpus = []
    for i in tqdm(range(len(df))):
        corpus.append(df.Phrase[i].lower().split())
    corpus = Counter(np.hstack(corpus))
    corpus = corpus
    corpus2 = sorted(corpus, key=corpus.get, reverse=True)
    print('Vocab ---- int')
    vocab_to_int = {word: idx for idx, word in enumerate(corpus2, 1)}
    print('phrase ---- int')
    phrase_to_int = []
    for i in tqdm(range(len(df))):
        phrase_to_int.append([vocab_to_int[word] for word in df.Phrase.values[i].lower().split()])
    return corpus, vocab_to_int, phrase_to_int


def Pad_sequences(phrase_to_int, seq_length):
    pad_sequences = np.zeros((len(phrase_to_int), seq_length), dtype=int)
    for idx, row in tqdm(enumerate(phrase_to_int), total=len(phrase_to_int)):
        pad_sequences[idx, :len(row)] = np.array(row)[:seq_length]
    return pad_sequences


class PhraseDataset(Dataset):
    def __init__(self, df, pad_sequences):
        super().__init__()
        self.df = df
        self.pad_sequences = pad_sequences

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if 'Sentiment' in self.df.columns:
            label = self.df['Sentiment'].values[idx]
            item = self.pad_sequences[idx]
            return item, label
        else:
            item = self.pad_sequences[idx]
            return item
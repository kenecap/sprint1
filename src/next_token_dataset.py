import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from collections import Counter

class TextDataset(Dataset):
    def __init__(self, file_path, vocab=None, max_vocab_size=20000):
        df = pd.read_csv(file_path)
        self.texts = df['text'].dropna().tolist()

        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        
        if vocab is None:
            self.word2idx = {
                self.pad_token: 0,
                self.unk_token: 1,
                self.bos_token: 2,
                self.eos_token: 3
            }
            word_counts = Counter(word for text in self.texts for word in text.split())
            most_common = word_counts.most_common(max_vocab_size - 4)
            for word, _ in most_common:
                self.word2idx[word] = len(self.word2idx)
        else:
            self.word2idx = vocab
        
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = text.split()
        unk_idx = self.word2idx[self.unk_token]
        
        seq = [self.word2idx[self.bos_token]]
        seq += [self.word2idx.get(token, unk_idx) for token in tokens]
        seq += [self.word2idx[self.eos_token]]

        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(seq[1:], dtype=torch.long)
        return input_seq, target_seq

class PadCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        inputs, targets = zip(*batch)
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad_idx)
        padded_targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        return padded_inputs, padded_targets
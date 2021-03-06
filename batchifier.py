import numpy as np
import random
import torch


class Batchifier(object):
    def __init__(self, idxs, pad_idx, batch_size, shuffle=False):
        self.idxs = idxs
        self.shuffle = shuffle

        self.batch_size = batch_size
        self.nbatches = len(idxs) // batch_size
        self.pad_idx = pad_idx
        self.current_idx = None
        self.batches = None
        self.reset()

    def reset(self):
        if self.shuffle:
            random.shuffle(self.idxs)
        self._build_data(self.idxs)
        self.current_idx = 0

    def _sort_batch(self, batch):
        if not isinstance(batch, np.ndarray):
            batch = np.array(batch)
        # line contains sos, eos tokens additionally
        lengths = np.array([len(line) - 1 for line in batch])
        sorted_idxs = np.argsort(lengths)[::-1]
        batch, lengths = batch[sorted_idxs], lengths[sorted_idxs]
        maxlen = lengths[0]

        src = np.full((self.batch_size, maxlen), self.pad_idx)
        tgt = np.copy(src)

        for idx, (line, length) in enumerate(zip(batch, lengths)):
            src[idx][:length] = line[:-1]
            tgt[idx][:length] = line[1:]

        src, tgt, lengths = torch.LongTensor(src), torch.LongTensor(tgt), torch.LongTensor(lengths)
        return src, tgt, lengths

    def _build_data(self, idxs):
        self.batches = []
        for batch_idx in range(self.nbatches):
            batch = idxs[batch_idx*self.batch_size: (batch_idx+1)*self.batch_size]
            src, tgt, lengths = self._sort_batch(batch)  # should sort batches to pass into lstm with padding
            self.batches.append((src, tgt, lengths))

    def __iter__(self):
        return self

    def __len__(self):
        return self.nbatches

    def __next__(self):
        try:
            result = self.batches[self.current_idx]
        except IndexError:
            self.reset()
            raise StopIteration
        self.current_idx += 1
        return result

    def get_random(self):
        return random.choice(self.batches)


import numpy as np
import random
import torch


class Batchifier(object):
    def __init__(self, idxs, pad_idx, batch_size, shuffle):
        if shuffle:
            raise RuntimeError("Not implemented yet")

        self.batch_size = batch_size
        self.nbatches = len(idxs) // batch_size
        self.pad_idx = pad_idx
        self.current_idx = 0
        self.batches = []
        self.build_data(idxs)

    def sort_batch(self, batch):
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
            src[idx][:length] = line[1:]
            tgt[idx][:length] = line[:-1]

        src, tgt, lengths = torch.LongTensor(src), torch.LongTensor(tgt), torch.LongTensor(lengths)
        return src, tgt, lengths

    def build_data(self, idxs):
        for batch_idx in range(self.nbatches):
            batch = idxs[batch_idx*self.batch_size: (batch_idx+1)*self.batch_size]
            src, tgt, lengths = self.sort_batch(batch)  # should sort batches to pass into lstm with padding
            self.batches.append((src, tgt, lengths))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = self.batches[self.current_idx]
        except IndexError:
            raise StopIteration
        self.current_idx += 1
        return result

    def get_random(self):
        return random.choice(self.batches)

import os
import logging
from utils import iget_line
from collections import Counter

logger = logging.getLogger()


class Dictionary(object):
    PAD = '<pad>'
    SOS = '<sos>'
    EOS = '<eos>'
    UNK = '<oov>'

    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}

        for special_token in self.special_tokens:
            self.add_token(special_token)

    def add_token(self, token):
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def __len__(self):
        return len(self.token2idx)

    def convert_token2idx(self, token):
        return self.token2idx.get(token, self.unk_idx)

    def convert_idx2token(self, idx):
        return self.idx2token[idx]

    def convert_tokens2idxs(self, raw_tokens):
        '''
        :param raw_tokens: list of tokens w/o special tokens
        :return: idxs: list of indexes
        '''
        return list(map(self.convert_token2idx, [self.sos] + raw_tokens + [self.eos]))

    def convert_idxs2tokens(self, idxs):
        return list(map(self.convert_idx2token, idxs))

    def convert_idxs2tokens_prettified(self, idxs):
        ''' The same as convert_idxs2tokens, but remove <sos>, <eos> and all after <eos>
        :param idxs: list of indexes to convert
        :return: list of tokens
        '''
        if not idxs:
            return None
        try:
            first_eos_idx = idxs.index(self.eos_idx)
        except ValueError:
            first_eos_idx = len(idxs)
        start_idx = 1 if idxs[0] == self.sos_idx else 0
        tokens = self.convert_idxs2tokens(idxs[start_idx:first_eos_idx])
        return tokens

    @property
    def pad(self):
        return self.PAD

    @property
    def pad_idx(self):
        return self.convert_token2idx(self.pad)

    @property
    def sos(self):
        return self.SOS

    @property
    def sos_idx(self):
        return self.convert_token2idx(self.sos)

    @property
    def eos(self):
        return self.EOS

    @property
    def eos_idx(self):
        return self.convert_token2idx(self.eos)

    @property
    def unk(self):
        return self.UNK

    @property
    def unk_idx(self):
        return self.convert_token2idx(self.unk)

    @property
    def special_tokens(self):
        return [self.PAD, self.SOS, self.EOS, self.UNK]


class Corpus(object):
    def __init__(self, path, n_tokens, sep=' '):
        self._sep = sep
        self.dictionary = Dictionary()
        self.maxlen = 0

        paths = {}
        self._labels = ['train', 'test']
        for label in self._labels:
            paths[label] = os.path.join(path, label + '.txt')

        logger.debug('Building dictionary')
        self.fill_dictionary(paths, n_tokens)
        logger.info('Dictionary size: {} (added special tokens)'.format(len(self.dictionary)))

        logger.debug('Tokenizing train dataset')
        self.train = self.tokenize(paths[self._labels[0]])
        logger.debug('Tokenizing test dataset')
        self.test = self.tokenize(paths[self._labels[1]])

    def fill_dictionary(self, paths, n_tokens):
        all_tokens = []
        for label, path in paths.items():
            for line in iget_line(path):
                tokens = line.split(self._sep)
                all_tokens += tokens
        counter = Counter(all_tokens)
        for token, _, in counter.most_common(n_tokens):
            self.dictionary.add_token(token)

    def tokenize(self, path):
        lines_idxs = []
        for line in iget_line(path):
            tokens = line.split(self._sep)
            idxs = self.dictionary.convert_tokens2idxs(tokens)
            lines_idxs.append(idxs)
            self.maxlen = max(self.maxlen, len(idxs))
        return lines_idxs

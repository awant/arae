import pytest
import tempfile
import os
from tempfile import TemporaryDirectory
from data import Dictionary, Corpus
from batchifier import Batchifier


def dump_sample_sentences():
    sents = [
        'The quick brown fox jumps over the lazy dog .',
        'I ate the dinner .',
        'I love learning .'
    ]
    n_tokens = 10

    tmpdir = TemporaryDirectory()
    with open(os.path.join(tmpdir.name, 'train.txt'), 'w', encoding='utf-8') as f:
        for sent in sents:
            f.write(sent+'\n')

    return sents, n_tokens, tmpdir


def test_dictionary():
    d = Dictionary()
    assert d.special_tokens == [d.PAD, d.SOS, d.EOS, d.UNK]
    assert d.convert_token2idx('<pad>') == 0
    assert d.convert_token2idx('<sos>') == 1
    assert d.convert_token2idx('<eos>') == 2
    assert d.convert_token2idx('<oov>') == 3

    sentence = "The quick brown fox jumps over the lazy dog"
    sent_ids = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 2]
    tokens = sentence.split(' ')
    for token in tokens:
        d.add_token(token)

    assert d.convert_tokens2idxs(tokens) == sent_ids
    assert ' '.join(d.convert_idxs2tokens_prettified(sent_ids)) == sentence


def test_corpus():
    sents, n_tokens, dir_ = dump_sample_sentences()
    tokens = [sent.split(' ') for sent in sents]
    corpus = Corpus(dir_.name, n_tokens)

    assert len(corpus.dictionary) == n_tokens
    assert corpus.maxlen == max(list(map(len, tokens)))+2  # +2 for <sos>, <eos>
    assert len(corpus.train) == len(sents)
    assert corpus.train[0][0] == corpus.dictionary.sos_idx
    assert corpus.train[0][-1] == corpus.dictionary.eos_idx
    assert corpus.test is None


def test_batchifier():
    sents, n_tokens, dir_ = dump_sample_sentences()
    tokens = [sent.split(' ') for sent in sents]
    corpus = Corpus(dir_.name, n_tokens)
    d = corpus.dictionary

    batchifier = Batchifier(corpus.train, d.pad_idx, batch_size=2)
    assert len(batchifier) == (len(sents) // 2)

    for src, tgt, lengths in batchifier:
        assert src[0][0] == d.sos_idx
        assert tgt[0][-1] == d.eos_idx
        assert ((src != d.pad_idx).long().sum(1) == lengths).all()
        assert ((tgt != d.pad_idx).long().sum(1) == lengths).all()


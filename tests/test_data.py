import pytest
import tempfile
import os
from tempfile import TemporaryDirectory
from data import Dictionary, Corpus


def dump_sample_sentences():
    sents = [
        'The quick brown fox jumps over the lazy dog .',
        'I ate the dinner .',
        'I love learning .'
    ]
    n_tokens = 10

    tmpdirname = TemporaryDirectory()
    with open(os.path.join(tmpdirname, 'train.txt'), 'w', encoding='utf-8') as f:
        for sent in sents:
            fd.write(sent+'\n')

    return sents, n_tokens, tmpdirname


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
    sents, n_tokens, path = dump_sample_sentences()
    corpus = Corpus(path, n_tokens)



import torch


def sample_noise(batch_size, internal_repr_size):
    return torch.Tensor(batch_size, internal_repr_size).normal_(0, 1)


def generate_sentences(autoencoder, generator, dictionary, count, maxlen, greedy=True):
    sos_idx = dictionary.sos_idx
    if count > 3000:
        raise RuntimeError("Batching is not implemented in generation yet")
    with torch.no_grad():
        noise = sample_noise(count, autoencoder.internal_repr_size)
        fake_repr = generator(noise)  # [B, H]
        lines = autoencoder.generate(fake_repr, sos_idx, maxlen, greedy)  # [B, L]
        lines = lines.cpu().numpy()

    convert_tokens2sents = lambda tokens: dictionary.convert_idxs2tokens_prettified(tokens)
    sentences = [convert_tokens2sents(tokens) for tokens in lines]
    return sentences

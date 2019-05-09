import torch
import numpy as np


def sample_noise(batch_size, internal_repr_size):
    return torch.Tensor(batch_size, internal_repr_size).normal_(0, 1)


def _generate_batch(autoencoder, generator, batch_size, sos_idx, maxlen, greedy=True):
    device = autoencoder.device

    with torch.no_grad():
        noise = sample_noise(batch_size, generator.inp_size).to(device)
        fake_repr = generator(noise)
        lines = autoencoder.generate(fake_repr, sos_idx, maxlen, greedy)
        lines = lines.cpu().numpy()
    return lines


def generate_sentences(autoencoder, generator, dictionary, count, maxlen, greedy=True, sep=' '):
    batch_size = 5000
    sos_idx = dictionary.sos_idx
    lines = []
    for batch_idx in range(0, count, batch_size):
        batch_s = min(batch_size, count - batch_idx)
        cur_lines = _generate_batch(autoencoder, generator, batch_s, sos_idx, maxlen, greedy)
        lines.append(cur_lines)
    lines = np.concatenate(lines, axis=1)

    convert_tokens2sents = lambda tokens: dictionary.convert_idxs2tokens_prettified(tokens)
    sentences = [sep.join(convert_tokens2sents(tokens)) for tokens in lines]
    return sentences


def dump_model(models, dictionary, opts, path):
    autoencoder, generator, discriminator = models
    checkpoint = {
        'autoencoder': autoencoder.state_dict(),
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'dictionary': dictionary,
        'opts': opts
    }
    torch.save(checkpoint, path)


def load_model(model_types, path, device):
    Autoencoder, Generator, Discriminator = model_types
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    dictionary = checkpoint['dictionary']
    opts = checkpoint['opts']
    autoencoder = Autoencoder.from_opts(opts)
    generator = Generator.from_opts(opts)
    discriminator = Discriminator.from_opts(opts)

    autoencoder.load_state_dict(checkpoint['autoencoder'], strict=False)
    generator.load_state_dict(checkpoint['generator'], strict=False)
    discriminator.load_state_dict(checkpoint['discriminator'], strict=False)

    autoencoder.to(device)
    generator.to(device)
    discriminator.to(device)

    return (autoencoder, generator, discriminator), dictionary, opts


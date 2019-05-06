import torch


def sample_noise(batch_size, internal_repr_size):
    return torch.Tensor(batch_size, internal_repr_size).normal_(0, 1)


def generate_sentences(autoencoder, generator, dictionary, count, maxlen, greedy=True, sep=' '):
    is_cuda = next(generator.parameters()).is_cuda
    device = torch.device('cuda' if is_cuda else 'cpu')
    sos_idx = dictionary.sos_idx
    if count > 3000:
        raise RuntimeError("Batching is not implemented in generation yet")
    with torch.no_grad():
        noise = sample_noise(count, generator.inp_size).to(device)
        fake_repr = generator(noise)  # [B, H]
        lines = autoencoder.generate(fake_repr, sos_idx, maxlen, greedy)  # [B, L]
        lines = lines.cpu().numpy()

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


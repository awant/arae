from opts import configure_args
from data import Corpus
from batchifier import Batchifier
from models import Seq2Seq, Gen, Critic
import torch
import torch.optim as optim
import torch.nn.functional as F
import logging


# Global sets
logger = logging.getLogger()
Autoencoder, Generator, Discriminator = Seq2Seq, Gen, Critic


def sample_noise(batch_size, internal_repr_size):
    return torch.Tensor(batch_size, internal_repr_size).normal_(0, 1)


def form_log_line(metrics, niters):
    return ''


def train_autoencoder(autoencoder, optim_ae, criterion_ae, batch, device):
    autoencoder.train()
    optim_ae.zero_grad()

    # src: [B, L], tgt: [B, L], lengths: [B]
    src, tgt, lengths = [obj.to(device) for obj in batch]
    out = autoencoder(src, lengths, noise=True)  # [B, L, V]
    plain_out = out.view(-1, out.size(-1))
    plain_tgt = tgt.view(-1)

    loss = criterion_ae(plain_out, plain_tgt)
    loss.backward()
    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)
    optim_ae.step()

    # compute metrics for evaluation
    nmatches = (plain_out.argmax(dim=-1) == plain_tgt).float() * (plain_tgt == criterion_ae.ignore_index)
    ntokens = lengths.sum()

    metrics = {
        'loss_ae': loss.cpu(),
        'ntokens': ntokens,
        'nmatches': nmatches
    }
    return metrics


def calc_gradient_penalty(discriminator, real_data, fake_data, gan_gp_lambda, device):
    batch_size, h_size = real_data.size()
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, h_size).to(device)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    disc_interpolates = discriminator(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gan_gp_lambda
    return gradient_penalty


def train_discriminator(models, optim_discr, batch, gan_gp_lambda, device):
    autoencoder, generator, discriminator = models
    autoencoder.eval()
    generator.eval()
    discriminator.train()
    optim_discr.zero_grad()

    # src: [B, L], tgt: [B, L], lengths: [B]
    src, tgt, lengths = [obj.to(device) for obj in batch]
    batch_size = src.size(0)
    internal_repr_size = autoencoder.internal_repr_size

    noise = sample_noise(batch_size, internal_repr_size)
    with torch.no_grad():
        real_repr = autoencoder.encode(src, lengths, noise=False)  # [B, H]
        fake_repr = generator(noise)  # [B, H]

    real_discr_out = discriminator(real_repr.detach())  # [B, 1]
    fake_discr_out = discriminator(fake_repr.detach())
    (real_discr_out - fake_discr_out).backward()

    gradient_penalty = calc_gradient_penalty(discriminator, real_repr, fake_repr, gan_gp_lambda, device)
    gradient_penalty.backward()

    optim_discr.step()

    metrics = {
        'discr_real_loss': real_discr_out.mean().cpu(),
        'discr_fake_loss': fake_discr_out.mean().cpu()
    }
    return metrics


def train_discr_autoencoder(autoencoder, optim_ae, batch, grad_lambda, clip):
    autoencoder.train()
    discriminator.eval()
    optim_ae.zero_grad()

    # src: [B, L], tgt: [B, L], lengths: [B]
    src, tgt, lengths = [obj.to(device) for obj in batch]

    real_repr = autoencoder.encode(src, lengths, noise=False)
    real_repr.register_hook(lambda grad: grad * grad_lambda)

    neg_real_discr_out = -discriminator(real_repr)
    neg_real_discr_out.backward()
    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), clip)

    optim_ae.step()
    return -neg_real_discr_out.mean().cpu()


def train_generator(generator, discriminator, optim_gen, batch_size):
    generator.train()
    discriminator.eval()
    optim_gen.zero_grad()

    internal_repr_size = generator.inp_size
    noise = sample_noise(batch_size, internal_repr_size)
    fake_repr = generator(noise)
    fake_discr_out = discriminator(fake_repr)
    fake_discr_out.backward()
    optim_gen.step()

    metrics = {
        'gen_loss': fake_repr.mean().cpu()
    }

    return metrics


def train_epoch(models, optimizers, criterions, batchifier, args, device):
    autoencoder, generator, discriminator = models
    optim_ae, optim_gen, optim_discr = optimizers
    criterion_ae, = criterions

    niters_autoencoder = args.niters_ae
    niters_gan = 1  # hardcoded for now
    niters_discriminator = args.niters_gan_d
    niters_discr_autoencoder = args.niters_gan_ae
    niters_generator = args.niters_gan_g

    anneal_noise_every = 100  # hardcoded for now

    for batch_idx, batch in enumerate(batchifier, start=1):
        metrics = train_autoencoder(autoencoder, optim_ae, batch, device)

        if batch_idx % niters_autoencoder == 0:  # training GAN part
            for _ in range(niters_gan):
                for _ in range(niters_discriminator):
                    discr_metrics = train_discriminator(models, optim_discr, batchifier.get_random(),
                                                        args.gan_gp_lambda, device)
                    metrics.update(discr_metrics)
                for _ in range(niters_discr_autoencoder):
                    train_discr_autoencoder(autoencoder, optim_ae, batchifier.get_random(), args.grad_lambda, args.clip)
                for _ in range(niters_generator):
                    gen_metrics = train_generator(generator, discriminator, optim_gen, args.batch_size)
                    metrics.update(gen_metrics)
        if batch_idx % args.print_every == 0:
            line = form_log_line(metrics, args.print_every)
            logger.info(line)

        if batch_idx % anneal_noise_every == 0:
            autoencoder.noise_anneal(args.noise_anneal)


def evaluate(models, batchifier):
    autoencoder, generator, discriminator = models
    autoencoder.eval()
    generator.eval()
    discriminator.eval()

    ...


def train(models, optimizers, criterions, train_batchifier, test_batchifier, args, device):
    for epoch_idx in range(1, args.epochs+1):
        train_epoch(models, optimizers, criterions, train_batchifier, args, device)
        evaluate(models, test_batchifier)


if __name__ == '__main__':
    args = configure_args()
    device = torch.device('cuda' if args.gpu > -1 else 'cpu')

    # Data
    corpus = Corpus(args.data, n_tokens=args.vocab_size)
    vocab = corpus.dictionary
    pad_idx = vocab.pad_idx

    train_batchifier = Batchifier(corpus.train, pad_idx, args.batch_size)
    test_batchifier = Batchifier(corpus.test, pad_idx, args.batch_size)

    # Model
    autoencoder = Autoencoder.from_opts(args).to(device)
    generator = Generator.from_opts(args).to(device)
    discriminator = Discriminator.from_opts(args).to(device)
    models = (autoencoder, generator, discriminator)

    # Optimizers
    optim_ae = optim.SGD(autoencoder.parameters(), lr=args.lr_ae)
    optim_gen = optim.Adam(generator.parameters(), lr=args.lr_gan_g, betas=(args.beta1, 0.999))
    optim_discr = optim.Adam(discriminator.parameters(), lr=args.lr_gan_d, betas=(args.beta1, 0.999))
    optimizers = (optim_ae, optim_gen, optim_discr)

    # Criterions
    criterion_ae = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    criterions = (criterion_ae,)

    train(models, optimizers, criterions, train_batchifier, test_batchifier, args, device)

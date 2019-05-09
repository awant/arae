from opts import configure_args
from data import Corpus
from batchifier import Batchifier
from models import Seq2Seq, Gen, Critic, KenlmModel
from models import sample_noise, generate_sentences, dump_model
import torch
from torch import nn
import torch.optim as optim
import logging
from utils import Metrics
import math

# Global sets
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

Autoencoder, Generator, Discriminator = Seq2Seq, Gen, Critic


def form_log_line(metrics):
    metrics['ppl'] = math.exp(metrics['loss_ae'] / metrics['niter_clear'])  # wrong metric
    metrics['acc'] = metrics['nmatches'] / metrics['ntokens'] * 100
    metrics['loss_d'] = metrics['discr_loss'] / metrics['niter_clear']
    metrics['loss_g'] = metrics['gen_loss'] / metrics['niter_clear']
    line = '[{epoch:3d}/{nepoch:3d}][{iter:5d}/{niter:5d}] | ' \
           'ppl {ppl:8.2f} | acc {acc:4.2f} | '\
           'loss_d {loss_d:.3f} | loss_g {loss_g:.3f}'.format(**metrics)
    return line


def form_eval_log_line(metrics):
    metrics['ppl'] = math.exp(metrics['loss_ae'] / metrics['niter_clear'])  # wrong metric
    metrics['acc'] = metrics['nmatches'] / metrics['ntokens'] * 100
    line = '[{epoch:3d}/{nepoch:3d}] | test ppl {ppl:8.2f} | test acc {acc:4.2f}'
    if metrics['forward_ppl'] > 0:
        line += ' | forward ppl {forward_ppl:8.2f}'
    if metrics['reverse_ppl'] > 0:
        line += ' | reverse ppl {reverse_ppl:8.2f}'
    return line.format(**metrics)


def batches_to_sentences(batchifier, dictionary):
    sentences = []
    for src, _, _ in batchifier:
        sentences += [dictionary.convert_idxs2tokens_prettified(line) for line in src]
    return sentences


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
    nn.utils.clip_grad_norm_(autoencoder.parameters(), args.clip)
    optim_ae.step()

    # compute metrics for evaluation
    nmatches = (plain_out.argmax(dim=-1) == plain_tgt).float() * (plain_tgt != criterion_ae.ignore_index).float()
    nmatches = nmatches.sum().cpu().item()
    ntokens = lengths.sum().cpu().item()

    metrics = {
        'loss_ae': loss.cpu().item(),
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
    discriminator.train()
    optim_discr.zero_grad()

    # src: [B, L], tgt: [B, L], lengths: [B]
    src, tgt, lengths = [obj.to(device) for obj in batch]
    batch_size = src.size(0)

    noise = sample_noise(batch_size, generator.inp_size).to(device)
    with torch.no_grad():
        real_repr = autoencoder(src, lengths, encode_only=True, noise=False)  # [B, H]
    fake_repr = generator(noise)  # [B, H]

    real_discr_out = discriminator(real_repr.detach())  # [B, 1]
    fake_discr_out = discriminator(fake_repr.detach())
    discr_loss = (real_discr_out - fake_discr_out).mean()

    gradient_penalty = calc_gradient_penalty(discriminator, real_repr, fake_repr, gan_gp_lambda, device)
    (discr_loss + gradient_penalty).backward()  # basically: WGAN-GP

    optim_discr.step()

    metrics = {
        'discr_loss': discr_loss.cpu().item(),
    }
    return metrics

def train_encoder_by_discriminator(autoencoder, optim_ae, batch, grad_lambda, clip):
    autoencoder.train()
    optim_ae.zero_grad()

    # src: [B, L], tgt: [B, L], lengths: [B]
    src, tgt, lengths = [obj.to(device) for obj in batch]

    real_repr = autoencoder(src, lengths, encode_only=True, noise=False)
    real_repr.register_hook(lambda grad: grad * grad_lambda)

    discr_out = discriminator(real_repr)
    discr_loss = -discr_out.mean()
    discr_loss.backward()
    nn.utils.clip_grad_norm_(autoencoder.parameters(), clip)

    optim_ae.step()
    return -discr_loss.cpu().item()


def train_generator(generator, discriminator, optim_gen, batch_size, device):
    generator.train()
    optim_gen.zero_grad()

    noise = sample_noise(batch_size, generator.inp_size).to(device)
    fake_repr = generator(noise)
    fake_discr_out = discriminator(fake_repr)
    fake_discr_loss = fake_discr_out.mean()
    fake_discr_loss.backward()
    optim_gen.step()

    metrics = {
        'gen_loss': fake_discr_loss.cpu().item()
    }
    return metrics


def train_epoch(models, optimizers, criterions, batchifier, args, epoch_idx, device):
    autoencoder, generator, discriminator = models
    optim_ae, optim_gen, optim_discr = optimizers
    criterion_ae, = criterions

    niters_autoencoder = args.niters_ae
    niters_gan = 1  # hardcoded for now
    niters_discriminator = args.niters_gan_d
    niters_discr_autoencoder = args.niters_gan_ae
    niters_generator = args.niters_gan_g

    anneal_noise_every = 100  # hardcoded for now

    def init_metrics():
        return Metrics(
            epoch=epoch_idx,
            nepoch=args.epochs,
            niter=len(batchifier),
            niter_clear=args.print_every
        )

    metrics = init_metrics()
    for batch_idx, batch in enumerate(batchifier, start=1):
        metrics['iter'] = batch_idx
        metrics_ae = train_autoencoder(autoencoder, optim_ae, criterion_ae, batch, device)
        metrics.accum(metrics_ae)
        if batch_idx % niters_autoencoder == 0:  # training GAN part
            for _ in range(niters_gan):
                for _ in range(niters_discriminator):
                    random_batch = batchifier.get_random()
                    # loss = (D(enc(src)) - D(G(noise))).mean() + GP
                    discr_metrics = train_discriminator(models, optim_discr, random_batch,
                                                        args.gan_gp_lambda, device)
                    metrics.accum(discr_metrics)
                for _ in range(niters_discr_autoencoder):
                    random_batch = batchifier.get_random()
                    # loss = -D(enc(src))
                    train_encoder_by_discriminator(autoencoder, optim_ae, random_batch, args.grad_lambda,
                                                   args.clip)
                for _ in range(niters_generator):
                    # loss = D(G(noise))
                    gen_metrics = train_generator(generator, discriminator, optim_gen, args.batch_size, device)
                    metrics.accum(gen_metrics)
        if batch_idx % args.print_every == 0:
            line = form_log_line(metrics)
            logger.info(line)
            metrics = init_metrics()

        if batch_idx % anneal_noise_every == 0:
            autoencoder.noise_anneal(args.noise_anneal)


def evaluate(models, criterions, batchifier, dictionary, args, epoch_idx, kenlm, kenlm_eval_size=100000, maxlen=15):
    show_reconstructions = True

    autoencoder, generator, discriminator = models
    criterion_ae, = criterions
    autoencoder.eval()
    generator.eval()
    discriminator.eval()

    metrics = Metrics(
        epoch=epoch_idx,
        nepoch=args.epochs,
        niter_clear=len(batchifier)
    )
    with torch.no_grad():
        for batch in batchifier:
            src, tgt, lengths = [obj.to(device) for obj in batch]

            out = autoencoder(src, lengths, noise=False)
            plain_out, plain_tgt = out.view(-1, out.size(-1)), tgt.view(-1)
            metrics['loss_ae'] += criterion_ae(plain_out, plain_tgt).cpu().item()

            nmatches = (plain_out.argmax(dim=-1) == plain_tgt).float()*(plain_tgt != criterion_ae.ignore_index).float()
            metrics['nmatches'] += nmatches.sum().cpu().item()
            metrics['ntokens'] += lengths.sum().cpu().item()

    if show_reconstructions:
        tgt_idxs, reconstr_idxs = tgt[:3].cpu().numpy(), out[:3].argmax(-1).cpu().numpy()
        tgt_tokens = list([dictionary.convert_idxs2tokens_prettified(x) for x in tgt_idxs])
        rec_tokens = list([dictionary.convert_idxs2tokens_prettified(x) for x in reconstr_idxs])
        tgt_tokens = '\n    '.join([' '.join(tokens) for tokens in tgt_tokens])
        rec_tokens = '\n    '.join([' '.join(tokens) for tokens in rec_tokens])
        logger.debug('Orig sents:\n    '+tgt_tokens+'\nReconstructions:\n    '+rec_tokens)

    if kenlm is not None:
        sentences = generate_sentences(autoencoder, generator, dictionary, count=kenlm_eval_size, maxlen=maxlen, greedy=True)
        logger.debug('Generated sentences:\n    '+'\n    '.join(sentences[:5]))
        metrics['forward_ppl'] = kenlm.get_ppl(sentences)
        gen_kenlm = KenlmModel.build(sentences)
        if gen_kenlm:
            test_sentences = batches_to_sentences(batchifier, dictionary)
            metrics['reverse_ppl'] = gen_kenlm.get_ppl(test_sentences)

    line = form_eval_log_line(metrics)
    logger.info(line)
    return metrics['ppl']


def train(models, dictionary, optimizers, criterions, train_batchifier, test_batchifier, args, kenlm, maxlen, device):
    best_ppl = 1e10
    for epoch_idx in range(1, args.epochs+1):
        train_epoch(models, optimizers, criterions, train_batchifier, args, epoch_idx, device)
        ppl = evaluate(models, criterions, test_batchifier, dictionary, args, epoch_idx, kenlm, maxlen)
        if ppl < best_ppl:
            best_ppl = ppl
            logger.info('Dump best model by far for {} epoch'.format(epoch_idx))
            dump_model(models, dictionary, args, args.save)


if __name__ == '__main__':
    args = configure_args()
    device = torch.device('cuda' if args.gpu > -1 else 'cpu')

    # Data
    logger.info('Building data...')
    corpus = Corpus(args.data, n_tokens=args.vocab_size)
    dictionary = corpus.dictionary
    pad_idx = dictionary.pad_idx
    maxlen = corpus.maxlen

    logger.info('Building batchifier...')
    train_batchifier = Batchifier(corpus.train, pad_idx, args.batch_size)
    test_batchifier = Batchifier(corpus.test, pad_idx, args.batch_size)

    # Model
    logger.info('Building models...')
    autoencoder = Autoencoder.from_opts(args).to(device)
    generator = Generator.from_opts(args).to(device)
    discriminator = Discriminator.from_opts(args).to(device)
    models = (autoencoder, generator, discriminator)
    # Kenlm model
    kenlm = None
    if args.kenlm_model:
        logger.info('Loading Kenlm model...')
        kenlm = KenlmModel(args.kenlm_model)

    # Optimizers
    optim_ae = optim.SGD(autoencoder.parameters(), lr=args.lr_ae)
    optim_gen = optim.Adam(generator.parameters(), lr=args.lr_gan_g, betas=(args.beta1, 0.999))
    optim_discr = optim.Adam(discriminator.parameters(), lr=args.lr_gan_d, betas=(args.beta1, 0.999))
    optimizers = (optim_ae, optim_gen, optim_discr)

    # Criterions
    criterion_ae = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    criterions = (criterion_ae,)

    logger.info('Training...')
    train(models, dictionary, optimizers, criterions, train_batchifier, test_batchifier, args, kenlm, maxlen, device)


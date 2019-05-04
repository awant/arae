import argparse


def configure_args():
    parser = argparse.ArgumentParser(description='Training ARAE model')
    parser.add_argument('--data_oneb', type=str, required=True,
                        help='path to preprocessed data_oneb corpus with train.txt, test.txt with space as separator')
    parser.add_argument('--save', type=str, required=True, help='path for best .pt checkpoint')

    parser.add_argument('--vocab_size', type=int, default=30000, help='vocabulary size')
    parser.add_argument('--emsize', type=int, default=500, help='size of embeddings')
    parser.add_argument('--nhidden', type=int, default=500, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
    parser.add_argument('--noise_r', type=float, default=0.1, help='std of noise for autoencoder')
    parser.add_argument('--noise_anneal', type=float, default=0.9995, help='anneal noise_r every 100 iterations')
    parser.add_argument('--hidden_init', action='store_true', help="initialize decoder hidden state with encoder's")
    parser.add_argument('--arch_g', type=str, default='500-500', help='generator architecture (MLP)')
    parser.add_argument('--arch_d', type=str, default='500-500', help='critic/discriminator architecture (MLP)')
    parser.add_argument('--z_size', type=int, default=100, help='dimension of random noise z to feed into generator')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout applied to layers')

    parser.add_argument('--epochs', type=int, default=15, help='maximum number of epochs')
    parser.add_argument('--no_earlystopping', action='store_true', help="won't use KenLM for early stopping")
    parser.add_argument('--patience', type=int, default=2, help="lm evaluations w/o increasing for early stopping")
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
    parser.add_argument('--niters_ae', type=int, default=1, help='number of autoencoder iterations in training')
    parser.add_argument('--niters_gan_d', type=int, default=5, help='number of discriminator iterations in training')
    parser.add_argument('--niters_gan_g', type=int, default=1, help='number of generator iterations in training')
    parser.add_argument('--niters_gan_ae', type=int, default=1, help='number of gan-into-ae iterations in training')
    parser.add_argument('--niters_gan_schedule', type=str, default='',
                        help='epoch counts to increase number of GAN training '
                             ' iterations (increment by 1 each time)')
    parser.add_argument('--lr_ae', type=float, default=1, help='autoencoder learning rate')
    parser.add_argument('--lr_gan_g', type=float, default=1e-04, help='generator learning rate')
    parser.add_argument('--lr_gan_d', type=float, default=1e-04, help='critic/discriminator learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--clip', type=float, default=1, help='gradient clipping, max norm')
    parser.add_argument('--gan_clamp', type=float, default=0.01, help='WGAN clamp')
    parser.add_argument('--gan_gp_lambda', type=float, default=10, help='WGAN GP penalty lambda')
    parser.add_argument('--grad_lambda', type=float, default=1, help='WGAN into AE lambda')

    parser.add_argument('--sample', action='store_true', help='sample when decoding for generation')
    parser.add_argument('--N', type=int, default=5, help='N-gram order for training n-gram language model')
    parser.add_argument('--log_interval', type=int, default=200, help='interval to log autoencoder training results')

    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    parser.add_argument('--kenlm_model', type=str, help='path to reference kenlm model for computing forward ppl')
    parser.add_argument('--gpu', type=int, default=-1, help='device to use. = -1 - don\'t use gpu')
    parser.add_argument('--print_every', type=int, default=100, help='show metrics for train dataset')

    return parser.parse_args()

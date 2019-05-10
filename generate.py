import argparse
from models import Seq2Seq, Gen, Critic
from models import load_model, generate_sentences
from utils import dump_lines
import torch


MODEL_TYPES = (Seq2Seq, Gen, Critic)


def parse_args():
    parser = argparse.ArgumentParser(description='Generating sentences with ARAE model')
    parser.add_argument('--checkpoint', type=str, default='lang_model.pt', help='path to .pt checkpoint')
    parser.add_argument('--count', type=int, default=1000, help='number of sentences to generate')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu index, -1 for cpu')
    parser.add_argument('--maxlen', type=int, default=15, help='max len of sent to generate')
    parser.add_argument('--out', type=str, default='generated.txt', help='filepath to generated sentences')
    parser.add_argument('--greedy', action='store_true', help='sentence generation')
    return parser.parse_args()


def main(args):
    device = torch.device('cuda' if args.gpu > -1 else 'cpu')
    (autoencoder, generator, dictionary), dictionary, opts = load_model(MODEL_TYPES, args.checkpoint, device)
    sentences = generate_sentences(autoencoder, generator, dictionary, args.count, maxlen=args.maxlen,
                                   greedy=args.greedy)
    dump_lines(args.out, sentences)


if __name__ == '__main__':
    args = parse_args()
    main(args)


import torch
from torch import nn
from torch.nn.functional import softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Seq2Seq(nn.Module):
    def __init__(self, emsize, nhidden, ntokens, nlayers, noise_r=0.2, dropout=0):
        super().__init__()
        self._noise_r = noise_r
        self.internal_repr_size = nhidden
        self._device = None
        self.enc_embedding = nn.Embedding(ntokens, emsize)
        self.dec_embedding = nn.Embedding(ntokens, emsize)
        self.encoder = nn.LSTM(input_size=emsize, hidden_size=nhidden, num_layers=nlayers, dropout=dropout,
                               batch_first=True)
        self.decoder = nn.LSTM(input_size=emsize + nhidden, hidden_size=nhidden, num_layers=1, dropout=dropout,
                               batch_first=True)
        self.linear = nn.Linear(nhidden, ntokens)

        self.init_weights()

    def noise_anneal(self, alpha):
        self._noise_r *= alpha

    @classmethod
    def from_opts(cls, opts):
        return cls(opts.emsize, opts.nhidden, opts.vocab_size, opts.nlayers, opts.noise_r, opts.dropout)

    def init_weights(self):
        initrange = 0.1
        self.enc_embedding.weight.data.uniform_(-initrange, initrange)
        self.dec_embedding.weight.data.uniform_(-initrange, initrange)

        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def forward(self, inp, lengths, encode_only=False, noise=False):
        internal_repr = self.encode(inp, lengths, noise)  # [B, H]
        if encode_only:
            return internal_repr
        out = self.decode(internal_repr, inp, lengths)
        return out

    def encode(self, inp, lengths, noise):
        '''Encoding internal representation of inputs
        :param inp: input tokens, size: [B, L]
        :param lengths: real lengths of lines, size: [B]
        :param noise: should add noise for out representation
        :return: internal representation of inp, size: [B, H]
        '''
        embs = self.enc_embedding(inp)  # [B, L, E]
        packed_embeddings = pack_padded_sequence(embs, lengths=lengths, batch_first=True)
        _, (h_n, c_n) = self.encoder(packed_embeddings)

        internal_repr = h_n[-1]  # getting last layer, size: [B, H]
        internal_repr = internal_repr / torch.norm(internal_repr, p=2, dim=1, keepdim=True)

        if noise:
            assert self._noise_r > 0
            gauss_noise = torch.normal(mean=torch.zeros_like(internal_repr), std=self._noise_r)
            internal_repr += gauss_noise

        return internal_repr

    def decode(self, internal_repr, inp, lengths):
        '''Decoding an internal representation into tokens probs
        :param internal_repr: internal representation of inp tokens, size: [B, H]
        :param maxlen: max length of lines
        :param inp: inp tokens in batch, size: [B, L]
        :param lengths: lengths of input tokens, size: [B]
        :return: unsoftmaxed probs
        '''
        maxlen = lengths.max()

        # copy the internal representation for every token in line
        hiddens = internal_repr.unsqueeze(1).repeat(1, maxlen, 1)  # size: [B, L, H]

        embs = self.dec_embedding(inp)  # size: [B, L, E2]
        augmented_embs = torch.cat([embs, hiddens], -1)  # concat, size: [B, L, E2+H]

        packed_embs = pack_padded_sequence(input=augmented_embs, lengths=lengths, batch_first=True)
        packed_output, _ = self.decoder(packed_embs)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)  # size: [B, L, H]

        decoded = self.linear(output.contiguous())  # size: [B, L, V]
        return decoded

    def sample_next_idxs(self, inp, greedy=True):
        ''' Sample next idxs
        :param inp: inp tokens probs, size: [B, V]
        :param greedy: choose with the max prob
        :return: tokens idxs, size: [B]
        '''
        if greedy:
            return torch.argmax(inp, -1)
        probs = softmax(inp, dim=-1)
        return probs.multinomial(1).squeeze(1)

    def generate(self, internal_repr, sos_idx, maxlen, greedy):
        batch_size = internal_repr.size(0)

        # prepare a tensor for generated idxs
        generated_idxs = torch.zeros(maxlen, batch_size, dtype=torch.long).to(self.device)  # [L, B]
        # set SOS as first token
        generated_idxs[0] = sos_idx

        state = (torch.zeros(1, batch_size, self.internal_repr_size).to(self.device),
                 torch.zeros(1, batch_size, self.internal_repr_size).to(self.device))
        for token_idx in range(maxlen-1):
            cur_tokens = generated_idxs[token_idx].unsqueeze(1)  # [B, 1]

            cur_embs = self.dec_embedding(cur_tokens)  # [B, 1, E2]
            inputs = torch.cat([cur_embs, internal_repr.unsqueeze(1)], -1)  # [B, 1, E2+H]

            output, state = self.decoder(inputs, state)  # output size: [B, 1, H]
            decoded = self.linear(output.squeeze(1))  # [B, V]
            generated_idxs[token_idx+1] = self.sample_next_idxs(decoded, greedy)  # [B]

        return generated_idxs.transpose(0,1)

    @property
    def device(self):
        # lazy insta
        if self._device:
            return self._device
        is_cuda = next(self.parameters()).is_cuda
        self._device = torch.device('cuda' if is_cuda else 'cpu')
        return self._device


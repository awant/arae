import math
import kenlm


class KenlmModel(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = kenlm.Model(model_path)

    def get_ppl(self, sentences):
        n_tokens, nll10 = 0, 0
        for sent in sentences:
            sent = sent.strip()
            nll10 += -self.model.score(sent)
            n_tokens += len(sent.split())
        ppl = math.pow(10., nll10 / n_tokens)
        return ppl

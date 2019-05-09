import os
import math
import subprocess
import kenlm
from tempfile import TemporaryDirectory
from utils import dump_lines
from batchifier import Batchifier


class KenlmModel(object):
    LMPLZ_PATH = '/usr/local/bin/lmplz'

    def __init__(self, model):
        self.model_path = None
        if isinstance(model, str):
            self.model_path = model
            model = kenlm.Model(model)
        self.model = model

    @classmethod
    def build(cls, sentences, **kwargs):
        # sentences: list of sents without <sos>, <eos> or a filepath
        tmpdir = TemporaryDirectory()
        gen_model_path = os.path.join(tmpdir.name, 'tmp_knlm.arpa')
        tmp_filepath = os.path.join(tmpdir.name, 'sents_tmp.txt')
        dump_lines(tmp_filepath, sentences)
        params = {
            'lmplzp': kwargs.get('lmplz_path', cls.LMPLZ_PATH),
            'N': 5, 'trainp': tmp_filepath,
            'modelp': gen_model_path,
            'S': kwargs.get('compressing_rate', 20)
        }
        template_cmd = '{lmplzp} -o {N} '
        if params['S']:
            template_cmd += '-S {S}% '
        template_cmd += '< {trainp} > {modelp}'
        cmd = template_cmd.format(**params)
        try:
            print('Building a model, cmd: {}'.format(cmd))
            subprocess.call(cmd, shell=True)
            print('Built kenlm model, N = {}, path: {}'.format(params['N'], params['modelp']))
            model = kenlm.Model(params['modelp'])
            return cls(model)
        except:
            print('Can\'t build kenlm model, probably because of lack of memory, try to change -S')
            return None

    def _get_sentences_ppl(self, sentences):
        assert isinstance(sentences, list)
        n_tokens, nll10 = 0, 0
        for sent in sentences:
            sent = sent.strip()
            nll10 += -self.model.score(sent)
            n_tokens += len(sent.split())
        ppl = math.pow(10., nll10 / n_tokens)
        return ppl

    def get_ppl(self, sentences):
        return self._get_sentences_ppl(sentences)


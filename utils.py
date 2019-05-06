from collections import defaultdict


def iget_line(filepath, encoding='utf-8'):
    with open(filepath, 'r', encoding=encoding) as f:
        for line in f:
            yield line.strip()


def dump_lines(filepath, lines, encoding='utf-8'):
    with open(filepath, 'w', encoding=encoding) as f:
        for line in lines:
            f.write(line+'\n')


class Metrics(defaultdict):
    def __init__(self, **kwargs):
        super().__init__(int)
        self._accum_dict(kwargs)

    def _accum_dict(self, d):
        for key, val in d.items():
            self[key] = self.get(key, 0) + val

    def accum(self, *args, **kwargs):
        for arg in args:
            if not isinstance(arg, dict):
                raise ValueError("Can process only dicts")
            self._accum_dict(arg)
        self._accum_dict(kwargs)


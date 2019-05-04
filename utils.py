from collections import defaultdict


def iget_line(filepath, encoding='utf-8'):
    with open(filepath, 'r', encoding=encoding) as f:
        for line in f:
            yield line.strip()


class Metrics(defaultdict):
    def _accum_dict(self, d):
        for key, val in d.items():
            self[key] = self.get(key, 0) + val

    def accum(self, *args, **kwargs):
        for arg in args:
            if not isinstance(arg, dict):
                raise ValueError("Can process only dicts")
            self._accum_dict(arg)
        self._accum_dict(kwargs)


# class Metrics(object):
#     def __init__(self, **kwargs):
#         self._d = defaultdict(float)
#         self.accum(**kwargs)
#
#     def _accum_dict(self, d):
#         for key, val in d.items():
#             self._d[key] += val
#
#     def accum(self, *args, **kwargs):
#         for arg in args:
#             if not isinstance(arg, dict):
#                 raise ValueError("Can process only dicts")
#             self._accum_dict(arg)
#         self._accum_dict(kwargs)
#
#     def __getitem__(self, key):
#         return self._d[key]
#
#     def __setitem__(self, key, value):
#         self._d[key] = value

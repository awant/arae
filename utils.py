
def iget_line(filepath, encoding='utf-8'):
    with open(filepath, 'r', encoding=encoding) as f:
        for line in f:
            yield line.strip()

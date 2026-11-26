from wsp.run import open_session


def main():
    s = open_session('.')
    codec = s.subject.codec_map()
    rows = [int(codec['e']), int(codec['n'])]
    for suffix, prediction in (
        ('weight', 'CER drops about 17 percentage points; most e/n substitutions disappear.'),
        ('bias', 'Little further CER change; restore the intercept associated with each detector.'),
    ):
        name = 'O_18.lin.' + suffix
        swapped = s.subject.tensors()[name][rows[::-1]].copy()
        h = s.hypothesis('The e and n output logits are transposed: exchange their ' + suffix + ' entries.',
                         prediction, 'set_rows')
        s.edit('set_rows', [name], hypothesis=h, index=rows, values=swapped)
        s.keep_unmeasured('Reciprocal confusions and two displaced head rows support restoring the child logits intact.')
    s.freeze()


if __name__ == '__main__':
    main()

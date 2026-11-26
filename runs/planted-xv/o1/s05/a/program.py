from wsp.run import open_session


def common_deletions(conf):
    # Count the C0 leading omissions if the table exposes ref/hyp pairs.
    counts = []
    rows = conf.items() if isinstance(conf, dict) else conf
    for row in rows:
        if isinstance(row, dict):
            r, p, n = row.get('ref'), row.get('hyp'), row.get('count')
        elif isinstance(row, (tuple, list)) and len(row) == 3:
            r, p, n = row
        elif (isinstance(row, (tuple, list)) and len(row) == 2
              and isinstance(row[0], (tuple, list)) and len(row[0]) == 2):
            (r, p), n = row
        else:
            continue
        if isinstance(r, str) and len(r) == 1 and r in ' enircsutm' and p in ('', None, '∅'):
            try:
                counts.append(int(n))
            except (TypeError, ValueError):
                continue
    return sum(counts)


def main():
    s = open_session('.')
    c0 = s.measure('C0')
    names = ['L_16.layer.' + param + suffix
             for suffix in ('', '_reverse')
             for param in ('weight_ih_l0', 'weight_hh_l0', 'bias_hh_l0')]
    before = common_deletions(c0.confusions)
    for gain, hypothesis, prediction in [
        (2.0, 'Both L_16 directions have half-strength gate drives.', 'CER down 0.08–0.16 absolute'),
        (1.5, 'A partial restoration may recover character peaks with less gate saturation.', 'CER down 0.02–0.09 absolute'),
    ]:
        h = s.hypothesis(hypothesis, prediction, 'block scale')
        s.edit('scale', names, factor=gain, hypothesis=h)
        m = s.measure(f'L_16 x{gain:g}')
        after = common_deletions(m.confusions)
        reason = (f'CER {c0.cer:.4f}->{m.cer:.4f}; collapsed {c0.collapsed}->{m.collapsed}; '
                  f'common deletions in reported confusions {before}->{after}')
        s.note(reason + '; confusion sample: ' + str(m.confusions)[:500])
        if m.cer < c0.cer and (m.collapsed > c0.collapsed or (before and after >= before)):
            s.revert('Lower CER but failed collapse/deletion check: ' + reason)
        elif s.keep(reason):
            break
    s.freeze()


if __name__ == '__main__':
    main()

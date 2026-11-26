from wsp.run import open_session
import numpy as np


def pairs(metrics):
    # Count only reported confusions. Preserve a literal space distinct from a gap.
    def char(x):
        return '' if x in (None, '', '∅') else x

    result = {}
    raw = metrics.confusions
    for item in (raw.items() if isinstance(raw, dict) else raw):
        if isinstance(item, dict):
            if 'ref' not in item or 'hyp' not in item:
                continue
            a, b = item['ref'], item['hyp']
            n = item.get('count', item.get('n', 0))
        elif len(item) == 3:
            a, b, n = item
        elif len(item) == 2 and isinstance(item[0], (tuple, list)) and len(item[0]) == 2:
            (a, b), n = item
        else:
            continue
        result[char(a), char(b)] = int(n)
    return result


def crossed(p):
    return p.get(('e', 'n'), 0) + p.get(('n', 'e'), 0)


def omitted(p, space):
    d = sum(n for (a, b), n in p.items() if b == '' and a != '' and (a == ' ') == space)
    i = sum(n for (a, b), n in p.items() if a == '' and b != '' and (b == ' ') == space)
    return d - i


def main():
    s = open_session('.')
    last = s.measure('C0')
    cm = s.subject.codec_map()
    w, b = 'O_18.lin.weight', 'O_18.lin.bias'
    idx = [int(cm['e']), int(cm['n'])]
    tensors = s.subject.tensors()

    def decide(m, targeted, label):
        nonlocal last
        if m.cer >= last.cer or m.collapsed > last.collapsed or not targeted:
            s.revert(label + ': failed CER, collapse or targeted confusion check')
            return False
        kept = s.keep(label + ': lower CER, no extra collapsed lines, targeted check passed')
        if kept:
            last = m
        return kept

    h = s.hypothesis('The learned e and n affine classifiers are assigned to each other.',
                     'CER down 15–18 percentage points; reciprocal substitutions largely disappear.',
                     'output-row transposition')
    s.edit('set_rows', [w, b], hypothesis=h, index=idx,
           per_tensor={k: {'values': tensors[k][idx[::-1]].copy()} for k in (w, b)})
    m = s.measure('H1 e/n restored')
    before, after = crossed(pairs(last)), crossed(pairs(m))
    s.note(f'Reported reciprocal substitutions: {before} -> {after}')
    # If no structured baseline pair was available, demand a CER drop comparable to the pack count.
    repaired = decide(m, after < before * 0.25 if before else m.cer < last.cer - 0.12, 'H1')
    if repaired:
        for space, label, threshold, row, step in ((True, 'H2 space', 20, cm[' '], 0.75),
                                                  (False, 'H3 blank', 40, 0, -0.5)):
            excess = omitted(pairs(last), space)
            if excess < threshold or s.budget_left < 1:
                continue
            h = s.hypothesis(label + ': deletion excess suggests a logit calibration error.',
                             'CER down 0.1–0.4 percentage points; smaller deletion excess.',
                             'single output bias offset')
            delta = np.zeros_like(s.subject.tensors()[b])
            delta[row] = step
            s.edit('add', [b], hypothesis=h, delta=delta)
            m = s.measure(label)
            decide(m, omitted(pairs(m), space) < excess, label)
    s.freeze()


if __name__ == '__main__':
    main()

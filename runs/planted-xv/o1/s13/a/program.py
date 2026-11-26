from wsp.run import open_session


def main():
    s = open_session('.')
    baseline = s.measure('C0')
    name = 'O_18.lin.weight'
    row = int(s.subject.codec_map()['n'])
    h = s.hypothesis('The n weight row has reversed polarity, producing n on background frames.',
                     'Absolute CER down by 0.60-0.72; n insertion burden and collapsed lines fall.',
                     'single output weight row sign reversal')
    values = -s.subject.tensors()[name][[row]]
    s.edit('set_rows', [name], hypothesis=h, index=[row], values=values)
    m = s.measure('restore n polarity')
    gain = baseline.cer - m.cer
    profile_changed = m.confusions != baseline.confusions
    s.note(f'Absolute CER gain {gain:.4f}; collapsed lines {baseline.collapsed} -> {m.collapsed}; '
           f'post-repair confusions: {m.confusions}')
    if gain > 0 and m.collapsed <= baseline.collapsed and profile_changed:
        kept = s.keep('CER fell, confusions changed, and collapse did not worsen; require paired evidence.')
        if kept and gain < 0.60:
            s.note('Gain smaller than predicted; the sign reversal accounts for only part of the error.')
    else:
        s.revert('The CER, collapse, and confusion checks did not jointly support the repair.')
    s.freeze()


if __name__ == '__main__':
    main()

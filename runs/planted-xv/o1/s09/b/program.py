from wsp.kraken_subject import KrakenSubject
from wsp.run import open_session

PARENT = '/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel'


def main():
    s = open_session('.')
    names = ['C_3.co.weight', 'C_3.co.bias']
    child = s.subject.tensors()
    if all(not child[name].any() for name in names):
        parent = KrakenSubject(PARENT)
        donor = parent.tensors()
        h = s.hypothesis(
            'The zero C_3 block removes all image-dependent features.',
            'Restoring the parent filters and biases should end collapse on most lines '
            'and reduce CER from 1.00 to roughly 0.10-0.30.',
            'single convolution block transplant',
        )
        s.edit('transplant', names, hypothesis=h,
               per_tensor={name: {'donor': donor[name]} for name in names})
        s.keep_unmeasured('C_3 was entirely zero and its neighboring channels remain '
                          'aligned to the donor. Preserve all other child weights.')
    else:
        s.note('C_3 does not match the reported erasure; no blind edit.')
    s.freeze()


if __name__ == '__main__':
    main()

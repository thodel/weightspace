#!/usr/bin/env python3
"""Stage 5 harness check 1 (docs/TROCR_STAGE5_PREREG.md section 10): the output of ``trocr_stage5_train.py --max-steps 0`` equals the subject. Compares every tensor
bit for bit (``Subject.digest`` and ``np.array_equal``), the tokenizer files (vocab, merges, tokenizer.json). The config and preprocessor_config of the copy are rewritten by the installed transformers (generation keys moved,
dtype, loss_type) and are only reported; the reading pins its own generation settings, and the practice reading of the copy is compared with the cached C0 separately. Usage: ``... trocr_stage5_zerocheck.py SUBJECT_DIR ZERO_DIR`` (CPU)."""
import filecmp
import json
import sys
from pathlib import Path

import numpy as np

from wsp.trocr_subject import TrocrSubject

a, b = Path(sys.argv[1]), Path(sys.argv[2])
sa, sb = TrocrSubject(a, device="cpu"), TrocrSubject(b, device="cpu")
ta, tb = sa.tensors(), sb.tensors()
bad = [n for n in ta if n not in tb or not np.array_equal(ta[n], tb[n])]
files = {f: filecmp.cmp(a / f, b / f, shallow=False) for f in ("vocab.json", "merges.txt", "tokenizer.json") if (a / f).exists()}
cfg = json.loads((a / "config.json").read_text()) == json.loads((b / "config.json").read_text())
print("tensors", len(ta), "unequal", len(bad), bad[:3], "| digest equal", sa.digest() == sb.digest(), "| tokenizer/preproc files identical", files, "| config equal (informational)", cfg)
sys.exit(0 if not bad and all(files.values()) and sa.digest() == sb.digest() else 2)

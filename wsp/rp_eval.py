"""The decode comparator of phase 5: the trainer's ``evaluate_qlora`` with one change,
``repetition_penalty`` passed to every ``generate`` call (value from
``WSP_REPETITION_PENALTY``). Runs in the trainer's venv; used through
``wsp.vlm_reference --repetition-penalty``. Everything else is the trainer's code.
"""
import os
import sys

from transformers.generation.utils import GenerationMixin

_PENALTY = float(os.environ["WSP_REPETITION_PENALTY"])
_generate = GenerationMixin.generate


def generate(self, *args, **kwargs):
    kwargs["repetition_penalty"] = _PENALTY
    return _generate(self, *args, **kwargs)


GenerationMixin.generate = generate

if __name__ == "__main__":
    from vlm_train_svc.evaluate_qlora import main
    print(f"wsp.rp_eval: repetition_penalty={_PENALTY} on every generate()", flush=True)
    sys.exit(main())

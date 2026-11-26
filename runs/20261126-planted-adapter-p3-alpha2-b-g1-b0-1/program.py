"""Blind weight-space repair program for pack q01, task (b) (budget 0).

Defect identified from pack.md:
- lora_alpha is 256 (scale = 4.0) vs healthy siblings' lora_alpha = 128 (scale = 2.0).
- Every one of the 128 LoRA modules has ||Delta W|| / sibling_median in [1.60, 1.85]
  (total ||Delta W|| = 234.99 vs siblings 127.35 and 135.65) with normal effective
  ranks (16.6 to 48.2), indicating a uniform 2x over-scaling that causes 29/32 pages
  to loop to the 3,072-token cap (CER 2.5029).
- Scaling all 128 lora_B.weight tensors by 0.5 halves Delta W everywhere, restoring
  the effective scale to lora_alpha = 128 (scale = 2.0).
"""
from wsp.run import open_session


def main() -> None:
    s = open_session(".")
    sub = s.subject

    b_tensors = [f"{mod}.lora_B.weight" for mod in sub.modules()]

    h1 = s.hypothesis(
        "All 128 adapter modules are uniformly over-scaled by 2x because lora_alpha is 256 "
        "(scale 4.0) instead of the sibling recipe's 128 (scale 2.0), giving ||Delta W|| ratios "
        "of 1.60-1.85 across every module with normal effective ranks and causing 29/32 pages "
        "to loop to the token cap. Scaling every lora_B.weight by 0.5 restores the effective "
        "lora_alpha=128 update scale across the entire adapter.",
        "CER drops by ~2.2 to 2.4 points (from 2.503 to ~0.10-0.30) and pages at the 3,072-token "
        "cap drop from 29/32 to <= 3/32.",
        "scale",
    )
    s.edit("scale", b_tensors, hypothesis=h1, factor=0.5)
    s.keep_unmeasured(
        "Uniform 2x lora_alpha over-scaling across all 128 modules (alpha=256 vs siblings' 128); "
        "scaling all lora_B.weight by 0.5 restores the exact effective scale of 2.0."
    )

    s.freeze()


if __name__ == "__main__":
    main()

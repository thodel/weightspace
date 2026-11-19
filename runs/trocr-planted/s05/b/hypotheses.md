model id: claude-sonnet-5-5; settings: default/low reasoning effort; tools: Read, Bash (cat/awk/python on pack text only), Write. No passes, no model loaded.

H1. enc.03.ffn.out.{weight,bias} are scaled by ~0.5 (norm ratio 0.5027 / 0.5000 to donor, rel. distance 0.667, cosine 0.996 / 1.000; every other tensor is within the healthy range). Edit: scale x2.0 on both. Predicted: CER drops clearly (several points, from 0.2346); the dominant space-deletion confusion should shrink.

Not edited: tied embedding (distance 0.527, as healthy ~0.5, no zero or tiny rows), no heads flagged, no zeros or non-finite values, encoder layers 9-11 only moved a little (fine-tuning pattern, not a defect).
